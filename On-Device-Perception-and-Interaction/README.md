
# Front-end-android-development

## Project Description

This project is an Android app that demonstrates CameraX functionality. You can run the project using **Android Studio**.

## Project Structure

```
front-end-android-development
│
└── VERSION016
    └── app
        └── src
            └── main
                └── java
                    └── com
                        └── plcoding
                            └── cameraxguide
                                └── MainActivity.kt
```

## How to Run

Open Android Studio (we use version `2024.2.1 Patch 3`).

Clone the `MainActivity.kt` and `CameraPreview.kt` file, the path is as follows:

```
./app/src/main/java/com/plcoding/cameraxguide/MainActivity.kt
```

Click the **Run** button and select a device or emulator to run the app.

## Port Forwarding and Real-Time Debugging

1. In **Android Studio**'s **cmd**, enter the following command to forward the port:

   ```bash
   adb reverse tcp:5000 tcp:5000
   ```

   This will map the local port 5000 to port 5000 on the Android device.

2. Enter the following command to use **scrcpy** to mirror the device screen to your computer for real-time debugging:

   ```bash
   scrcpy
   ```

