import java.util.UUID

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.phoneserver"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.phoneserver"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }


    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
    }

}


dependencies {

    // Ktor Server
    val ktorVersion = "3.2.3" // Correct, latest stable version
    implementation("io.ktor:ktor-server-core-jvm:$ktorVersion")
    implementation("io.ktor:ktor-server-cio-jvm:$ktorVersion")
    implementation("io.ktor:ktor-server-websockets-jvm:$ktorVersion")

    // ASR
    implementation("com.alphacephei:vosk-android:0.3.46@aar")

    implementation("net.java.dev.jna:jna:5.14.0@aar")

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}

//val voskModelAssetDir = "src/main/assets/vosk-model-small-en-us-0.15"
val voskModelAssetDir = "src/main/assets/vosk-model-en-us-0.22-lgraph"

tasks.register("generateVoskUuid") {
    doLast {
        val assetsDir = file(voskModelAssetDir)
        assetsDir.mkdirs()
        val uuidFile = file("$voskModelAssetDir/uuid")
        // Recreate on each build so swapping models forces re-unpack
        uuidFile.writeText(UUID.randomUUID().toString())
        println("Wrote Vosk UUID to: ${uuidFile.path}")
    }
}

// Make sure it's there before assets are packaged
tasks.named("preBuild").configure { dependsOn("generateVoskUuid") }