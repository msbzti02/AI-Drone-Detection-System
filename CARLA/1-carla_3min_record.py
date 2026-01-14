import carla
import cv2
import numpy as np
import time
import math
import os
import queue
from datetime import datetime
import traceback
RECORD_DURATION = 180  
OUTPUT_DIR = "carla_recording_3min"
VIDEO_FPS = 20
DRONE_ALTITUDE = 30
FLIGHT_RADIUS = 80

print("CARLA 3-MINUTE VIDEO RECORDER")
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱ Recording duration: {RECORD_DURATION} seconds (3 minutes)")
if os.path.exists(OUTPUT_DIR):
    import shutil
    print(f"Cleaning existing {OUTPUT_DIR}...")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print("\nConnecting to CARLA...")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map_name = world.get_map().name
    print(f"Connected to: {map_name}")
except Exception as e:
    print(f"Failed to connect to CARLA: {e}")
    print("\nMake sure CARLA simulator is running!")
    print("   Start it with: CarlaUE4.exe or ./CarlaUE4.sh")
    exit(1)

original_settings = world.get_settings()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / VIDEO_FPS
world.apply_settings(settings)
blueprint_library = world.get_blueprint_library()
spawned_actors = []
spawned_walkers = []
walker_controllers = []

try:
    print("\nSpawning objects for detection...")

    spawn_points = world.get_map().get_spawn_points()
    vehicle_bps = blueprint_library.filter('vehicle.*')
    for i in range(30):  
        if i < len(spawn_points):
            bp = np.random.choice(list(vehicle_bps))
            vehicle = world.try_spawn_actor(bp, spawn_points[i])
            if vehicle:
                spawned_actors.append(vehicle)
                vehicle.set_autopilot(True)
    print(f"Spawned {len(spawned_actors)} vehicles")

    walker_bps = blueprint_library.filter('walker.pedestrian.*')
    walker_controller_bp = blueprint_library.find('controller.ai.walker')

    for i in range(50):  
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_point.location = loc
            bp = np.random.choice(list(walker_bps))
            walker = world.try_spawn_actor(bp, spawn_point)
            if walker:
                spawned_walkers.append(walker)
                controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                walker_controllers.append(controller)
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                controller.set_max_speed(1.4)

    print(f"Spawned {len(spawned_walkers)} pedestrians")
    print("\nSetting up recording camera...")
    image_queue = queue.Queue()

    def save_image(data):
        image_queue.put(data)
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick', '0.0')
    center = carla.Location(0, 0, 0)
    if spawn_points:
        xs = [sp.location.x for sp in spawn_points[:30]]
        ys = [sp.location.y for sp in spawn_points[:30]]
        center = carla.Location(np.mean(xs), np.mean(ys), DRONE_ALTITUDE)

    init_trans = carla.Transform(
        carla.Location(center.x + FLIGHT_RADIUS, center.y, DRONE_ALTITUDE),
        carla.Rotation(pitch=-30, yaw=180)
    )
    camera = world.spawn_actor(camera_bp, init_trans)
    camera.listen(save_image)
    spawned_actors.append(camera)
    spectator = world.get_spectator()

    print(f"Camera ready (1920x1080 @ {VIDEO_FPS} FPS)")
    video_filename = f"{OUTPUT_DIR}/carla_drone_3min_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, VIDEO_FPS, (1920, 1080))

    print(f"Video output: {video_filename}")

    print("\nStarting 3-minute recording...")
    print("   Press Ctrl+C to stop early")
    frame_count = 0
    start_time = time.time()
    angle = 0
    angle_speed = 360 / (RECORD_DURATION * VIDEO_FPS) 
    world.tick()
    time.sleep(0.5)
    while not image_queue.empty():
        try:
            image_queue.get_nowait()
        except queue.Empty:
            break

    while True:
        elapsed = time.time() - start_time
        
        if elapsed >= RECORD_DURATION:
            print(f"\nRecording complete! ({RECORD_DURATION} seconds)")
            break
        rad = math.radians(angle)
        x = center.x + FLIGHT_RADIUS * math.cos(rad)
        y = center.y + FLIGHT_RADIUS * math.sin(rad)
        z = DRONE_ALTITUDE + 10 * math.sin(rad * 2)  
        yaw = angle + 180  
        pitch = -25 - 10 * math.sin(rad)  
        
        transform = carla.Transform(
            carla.Location(x, y, z),
            carla.Rotation(pitch=pitch, yaw=yaw)
        )
        camera.set_transform(transform)
        spectator.set_transform(transform)
        world.tick()
        try:
            image = image_queue.get(timeout=2.0)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            frame_bgr = array[:, :, :3][:, :, ::-1].copy()  
            time_str = f"Time: {int(elapsed//60):02d}:{int(elapsed%60):02d}"
            cv2.putText(frame_bgr, time_str, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame_bgr, f"Frame: {frame_count}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            video_writer.write(frame_bgr)
            if frame_count % 100 == 0:
                sample_file = f"{OUTPUT_DIR}/frame_{frame_count:05d}.jpg"
                cv2.imwrite(sample_file, frame_bgr)
            frame_count += 1
            angle += angle_speed
            if frame_count % (VIDEO_FPS * 10) == 0:
                remaining = RECORD_DURATION - elapsed
                print(f" {int(elapsed):3d}s recorded | {int(remaining):3d}s remaining | {frame_count} frames")
                
        except queue.Empty:
            print("Missed frame - retrying...")
            continue
        except Exception as e:
            print(f"Frame error: {e}")
            traceback.print_exc()
            continue

except KeyboardInterrupt:
    print("\n Recording stopped by user")
except Exception as e:
    print(f"\nError during recording: {e}")
    traceback.print_exc()
finally:
    print("\nCleaning up...")
    try:
        video_writer.release()
    except:
        pass
    
    try:
        world.apply_settings(original_settings)
    except:
        pass
    try:
        camera.stop()
    except:
        pass   
    for actor in spawned_actors:
        try:
            if actor.is_alive:
                actor.destroy()
        except:
            pass
    
    for controller in walker_controllers:
        try:
            controller.stop()
            controller.destroy()
        except:
            pass

    for walker in spawned_walkers:
        try:
            if walker.is_alive:
                walker.destroy()
        except:
            pass

total_time = time.time() - start_time if 'start_time' in dir() else 0

print(" RECORDING SUMMARY")

print(f"    Duration: {total_time:.1f} seconds")
print(f"    Frames recorded: {frame_count if 'frame_count' in dir() else 0}")
print(f"    Video file: {video_filename if 'video_filename' in dir() else 'N/A'}")
print(f"    Output folder: {OUTPUT_DIR}/")

