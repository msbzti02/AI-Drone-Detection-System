import airsim
import time
print("\nConnecting to AirSim")
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected!")
client.enableApiControl(True)
client.armDisarm(True)
print("Drone armed and ready!")
print("\n Taking off...")
client.takeoffAsync().join()
print(" Takeoff complete!")
print("\nFlying to 15 meters altitude")
client.moveToZAsync(-15, 5).join()
print("Reached 15m altitude!")
print("\nFlying square pattern at 15m")
print("  Moving forward")

client.moveToPositionAsync(20, 0, -15, 5).join()
print("  Moving right")

client.moveToPositionAsync(20, 20, -15, 5).join()
print("  Moving backward")

client.moveToPositionAsync(0, 20, -15, 5).join()
print("  Returning to start")

client.moveToPositionAsync(0, 0, -15, 5).join()
print(" Square pattern complete!")

print("\nMoving to 25m altitude")
client.moveToZAsync(-25, 5).join()

print("\n Flying larger pattern at 25m")

print("Forward 40m")
client.moveToPositionAsync(40, 0, -25, 7).join()

print("Right 40m")
client.moveToPositionAsync(40, 40, -25, 7).join()

print("Backward")
client.moveToPositionAsync(0, 40, -25, 7).join()

print("Left")
client.moveToPositionAsync(-40, 40, -25, 7).join()

print("Forward")
client.moveToPositionAsync(-40, 0, -25, 7).join()

print("Return to center")
client.moveToPositionAsync(0, 0, -25, 7).join()

print("Large pattern complete!")

print("\nLow altitude pass at 8m")
client.moveToZAsync(-8, 3).join()

print(" Slow forward pass")
client.moveToPositionAsync(30, 0, -8, 3).join()

print(" Slow right")
client.moveToPositionAsync(30, 30, -8, 3).join()

print(" Return")
client.moveToPositionAsync(0, 0, -10, 5).join()

print("\nHovering for 2 seconds")
client.hoverAsync().join()
time.sleep(2)

print("\nLanding")
client.landAsync().join()
print("Landed!")

client.armDisarm(False)
client.enableApiControl(False)
print("FLIGHT COMPLETE!")
print(" Press R in AirSim to stop recording!")

