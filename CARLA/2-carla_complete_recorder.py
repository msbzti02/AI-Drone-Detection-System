import carla
import cv2
import numpy as np
import time
import math
import os
import queue
import json
from datetime import datetime
OUTPUT_DIR = "carla_complete_dataset"
HOVER_TIME = 0.5  
APPROACH_DISTANCE = 6  
ALTITUDE_OFFSET = 4  # 


print("CARLA COMPLETE OBJECT DATASET RECORDER")
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import shutil
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
os.makedirs(f"{OUTPUT_DIR}/images")
os.makedirs(f"{OUTPUT_DIR}/by_category")
categories = ['cars', 'trucks', 'motorcycles', 'bicycles', 'emergency', 'pedestrians', 'props']
for cat in categories:
    os.makedirs(f"{OUTPUT_DIR}/by_category/{cat}")

print("\nConnecting to CARLA...")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"Connected to: {world.get_map().name}")
except Exception as e:
    print(f"Failed to connect: {e}")
    exit(1)

original_settings = world.get_settings()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
all_spawned = []

try:
    VEHICLES = {
        'cars': [
            'vehicle.audi.a2', 'vehicle.audi.etron', 'vehicle.audi.tt',
            'vehicle.bmw.grandtourer', 'vehicle.chevrolet.impala',
            'vehicle.dodge.charger_2020', 'vehicle.ford.mustang',
            'vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe_2020',
            'vehicle.mini.cooper_s_2021', 'vehicle.nissan.patrol_2021',
            'vehicle.tesla.model3', 'vehicle.toyota.prius',
        ],
        'trucks': [
            'vehicle.carlamotors.carlacola', 
            'vehicle.carlamotors.european_hgv',
            'vehicle.tesla.cybertruck',
        ],
        'motorcycles': [
            'vehicle.harley-davidson.low_rider',
            'vehicle.kawasaki.ninja',
            'vehicle.vespa.zx125',
            'vehicle.yamaha.yzf',
        ],
        'bicycles': [
            'vehicle.bh.crossbike',
            'vehicle.diamondback.century',
            'vehicle.gazelle.omafiets',
        ],
        'emergency': [
            'vehicle.carlamotors.firetruck',
            'vehicle.dodge.charger_police_2020',
            'vehicle.ford.ambulance',
            'vehicle.mitsubishi.fusorosa',  
        ],
    }
    
    PEDESTRIANS = [f'walker.pedestrian.{i:04d}' for i in range(1, 21)]  # First 20 types
    PROPS = [
        'static.prop.bench01',
        'static.prop.bench02',
        'static.prop.bin',
        'static.prop.streetbarrier',
        'static.prop.trafficcone01',
        'static.prop.trafficwarning',
    ]
    
    print("\n Spawning all objects...")
    objects_to_visit = []
    spawn_idx = 0
    for category, vehicle_list in VEHICLES.items():
        print(f"\n    Spawning {category}...")
        for vehicle_id in vehicle_list:
            if spawn_idx >= len(spawn_points):
                break
            try:
                bp = blueprint_library.find(vehicle_id)
                vehicle = world.try_spawn_actor(bp, spawn_points[spawn_idx])
                if vehicle:
                    all_spawned.append(vehicle)
                    objects_to_visit.append({
                        'actor': vehicle,
                        'type': vehicle_id,
                        'category': category,
                        'name': vehicle_id.split('.')[-1]
                    })
                    print(f"       {vehicle_id.split('.')[-1]}")
                    spawn_idx += 1
            except Exception as e:
                print(f"      Failed: {vehicle_id} - {e}")
    
    print(f"\n    Spawning pedestrians...")
    walker_controller_bp = blueprint_library.find('controller.ai.walker')
    
    for ped_id in PEDESTRIANS:
        try:
            bp = blueprint_library.find(ped_id)
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc:
                spawn_point.location = loc
                walker = world.try_spawn_actor(bp, spawn_point)
                if walker:
                    all_spawned.append(walker)
                    # Add controller
                    controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                    all_spawned.append(controller)
                    controller.start()
                    controller.set_max_speed(0)  
                    
                    objects_to_visit.append({
                        'actor': walker,
                        'type': ped_id,
                        'category': 'pedestrians',
                        'name': ped_id.split('.')[-1]
                    })
                    print(f"       {ped_id.split('.')[-1]}")
        except Exception as e:
            print(f"       Failed: {ped_id}")
    
    print(f"\n    Spawning props...")
    for i, prop_id in enumerate(PROPS):
        try:
            bp = blueprint_library.find(prop_id)
            if i < len(spawn_points):
                prop_location = spawn_points[i].location
                prop_location.x += 3  # Offset from road
                prop_transform = carla.Transform(prop_location)
                prop = world.try_spawn_actor(bp, prop_transform)
                if prop:
                    all_spawned.append(prop)
                    objects_to_visit.append({
                        'actor': prop,
                        'type': prop_id,
                        'category': 'props',
                        'name': prop_id.split('.')[-1]
                    })
                    print(f"       {prop_id.split('.')[-1]}")
        except Exception as e:
            print(f"       Failed: {prop_id}")
    print(f"\n    Total objects spawned: {len(objects_to_visit)}")
    
    for _ in range(20):
        world.tick()
    
    
    print("\n Setting up drone camera...")
    image_queue = queue.Queue()
    def capture_image(data):
        image_queue.put(data)
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '90')
    init_loc = carla.Location(0, 0, 20)
    camera = world.spawn_actor(camera_bp, carla.Transform(init_loc))
    camera.listen(capture_image)
    all_spawned.append(camera)
    
    spectator = world.get_spectator()
    print("    Camera ready!")
    print(f"\n Visiting {len(objects_to_visit)} objects...")
    print("="*70)
    captured_images = []
    for idx, obj_info in enumerate(objects_to_visit):
        actor = obj_info['actor']
        obj_type = obj_info['type']
        category = obj_info['category']
        name = obj_info['name']
        try:
            transform = actor.get_transform()
            obj_loc = transform.location
        except:
            print(f"    [{idx+1}] Skipping {name} - no transform")
            continue
        
        print(f"    [{idx+1}/{len(objects_to_visit)}] {category}/{name}")
        
        angle = np.random.uniform(0, 360)
        rad = math.radians(angle)
        
        cam_x = obj_loc.x + APPROACH_DISTANCE * math.cos(rad)
        cam_y = obj_loc.y + APPROACH_DISTANCE * math.sin(rad)
        cam_z = obj_loc.z + ALTITUDE_OFFSET
        
        dx = obj_loc.x - cam_x
        dy = obj_loc.y - cam_y
        yaw = math.degrees(math.atan2(dy, dx))
        
        camera_transform = carla.Transform(
            carla.Location(cam_x, cam_y, cam_z),
            carla.Rotation(pitch=-20, yaw=yaw)
        )
        camera.set_transform(camera_transform)
        spectator.set_transform(camera_transform)
        for _ in range(int(HOVER_TIME * 20)):
            world.tick()
        while not image_queue.empty():
            try:
                image_queue.get_nowait()
            except:
                break
        world.tick()
        try:
            image = image_queue.get(timeout=2.0)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            frame = array[:, :, :3][:, :, ::-1].copy()
            filename = f"{idx+1:03d}_{category}_{name}.jpg"
            cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}", frame)
            cv2.imwrite(f"{OUTPUT_DIR}/by_category/{category}/{filename}", frame)
            captured_images.append({
                'index': idx + 1,
                'filename': filename,
                'category': category,
                'name': name,
                'type': obj_type,
                'location': {'x': obj_loc.x, 'y': obj_loc.y, 'z': obj_loc.z}
            })
            
            print(f"       Captured!")
            
        except Exception as e:
            print(f"       Capture failed: {e}")
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'total_objects': len(objects_to_visit),
        'captured_images': len(captured_images),
        'categories': {cat: len([x for x in captured_images if x['category'] == cat]) for cat in categories},
        'images': captured_images
    }
    with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(" CAPTURE SUMMARY")
    print(f"   Total objects: {len(objects_to_visit)}")
    print(f"   Images captured: {len(captured_images)}")
    print(f"\n   By category:")
    for cat in categories:
        count = len([x for x in captured_images if x['category'] == cat])
        if count > 0:
            print(f"      {cat}: {count}")
    
    print(f"\n    Images saved to: {OUTPUT_DIR}/images/")
    print(f"    By category: {OUTPUT_DIR}/by_category/")

except KeyboardInterrupt:
    print("\n Interrupted by user")

except Exception as e:
    print(f"\n Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\nCleaning up...")
    try:
        world.apply_settings(original_settings)
    except:
        pass
    for actor in all_spawned:
        try:
            if hasattr(actor, 'stop'):
                actor.stop()
            actor.destroy()
        except:
            pass
    
    print(" Cleanup complete!")
print(" DATASET CAPTURE COMPLETE!")
print(f"\n Output folder: {OUTPUT_DIR}/")
print(" Run YOLOv8x detection next!")

