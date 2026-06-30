plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.simpleserver"
    compileSdk = 36


    packaging {
        resources {
            // Minimum needed to fix your error:
            excludes += "META-INF/INDEX.LIST"

            // (Optional) Common extras that sometimes collide with Netty/Logback:
            excludes += "META-INF/DEPENDENCIES"
            excludes += "META-INF/LICENSE"
            excludes += "META-INF/LICENSE.txt"
            excludes += "META-INF/NOTICE"
            excludes += "META-INF/NOTICE.txt"
            excludes += "META-INF/io.netty.versions.properties"
        }
    }


    defaultConfig {
        applicationId = "com.example.simpleserver"
        // https://stackoverflow.com/questions/78175876/android-methodhandle-invoke-and-methodhandle-invokeexact-are-only-supported-sta
        minSdk = 26
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
        isCoreLibraryDesugaringEnabled = true
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

val ktor_version = "3.2.3" // The latest version
//val logback_version = "1.5.6" // Latest version of Logback
val logback_version = "1.4.14" // Last stable version of the 1.4.x series

dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.0.4")

    implementation("io.ktor:ktor-server-core-jvm:${ktor_version}")
//    implementation("io.ktor:ktor-server-netty-jvm:${ktor_version}")
    implementation("io.ktor:ktor-server-cio:${ktor_version}")
    implementation("io.ktor:ktor-server-websockets-jvm:${ktor_version}")

    implementation("ch.qos.logback:logback-classic:${logback_version}")

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