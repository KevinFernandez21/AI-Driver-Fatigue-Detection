#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
//
// WARNING!!! PSRAM IC required for UXGA resolution and high JPEG quality
//            Ensure ESP32 Wrover Module or other board with PSRAM is selected
//            Partial images will be transmitted if image exceeds buffer size
//
//            You must select partition scheme from the board menu that has at least 3MB APP space.
//            Face Recognition is DISABLED for ESP32 and ESP32-S2, because it takes up from 15
//            seconds to process single frame. Face Detection is ENABLED if PSRAM is enabled as well

// ===================
// Select camera model
// ===================
//#define CAMERA_MODEL_WROVER_KIT // Has PSRAM
//define CAMERA_MODEL_ESP_EYE  // Has PSRAM
//#define CAMERA_MODEL_ESP32S3_EYE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_PSRAM // Has PSRAM
//#define CAMERA_MODEL_M5STACK_V2_PSRAM // M5Camera version B Has PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_ESP32CAM // No PSRAM
//#define CAMERA_MODEL_M5STACK_UNITCAM // No PSRAM
//#define CAMERA_MODEL_M5STACK_CAMS3_UNIT  // Has PSRAM
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
//#define CAMERA_MODEL_TTGO_T_JOURNAL // No PSRAM
//#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM
// ** Espressif Internal Boards **
//#define CAMERA_MODEL_ESP32_CAM_BOARD
//#define CAMERA_MODEL_ESP32S2_CAM_BOARD
//#define CAMERA_MODEL_ESP32S3_CAM_LCD
//#define CAMERA_MODEL_DFRobot_FireBeetle2_ESP32S3 // Has PSRAM
//#define CAMERA_MODEL_DFRobot_Romeo_ESP32S3 // Has PSRAM
#include "camera_pins.h"

// ===========================
// Enter your WiFi credentials
// ===========================
const char *ssid = "NETLIFE-SANCHEZ";
const char *password = "kd200421";
const char* serverUrl = "https://backend-600343716837.southamerica-east1.run.app/predict";
void startCameraServer();
void setupLedFlash(int pin);

// Configuración del pin para el buzzer
const int buzzerPin = 14;

unsigned long lastClosedTime = 0;  // Momento en que se detectaron ojos cerrados por última vez
bool alarmActive = false;         // Estado de la alarma

// Duración requerida para activar la alarma (6 segundos)
const unsigned long alarmThreshold = 6000; 

String sendImageToServer(uint8_t* image_buffer, size_t len) {
  const char* host = "192.168.100.76";  // Dirección del servidor
  const int port = 8000;                // Puerto del servidor

  WiFiClient client;

  // Intentar conectar al servidor
  if (!client.connect(host, port)) {
    Serial.println("❌ Error al conectar al servidor");
    return "";
  }

  String esp32ID = String(ESP.getEfuseMac(), HEX);  // Obtener ID único del ESP32
  String boundary = "----WebKitFormBoundary";       // Delimitador para multipart/form-data

  // Crear la cabecera HTTP
  String request = "POST /predict HTTP/1.1\r\n";
  request += "Host: " + String(host) + "\r\n";
  request += "Content-Type: multipart/form-data; boundary=" + boundary + "\r\n";
  
  // Crear la parte del cuerpo con la ID del ESP32
  String bodyStart = "--" + boundary + "\r\n";
  bodyStart += "Content-Disposition: form-data; name=\"esp32_id\"\r\n\r\n";
  bodyStart += esp32ID + "\r\n";

  // Adjuntar la imagen
  String imageHeader = "--" + boundary + "\r\n";
  imageHeader += "Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n";
  imageHeader += "Content-Type: image/jpeg\r\n\r\n";

  // Crear el cierre del multipart/form-data
  String bodyEnd = "\r\n--" + boundary + "--\r\n";

  // Calcular el tamaño total de la solicitud
  size_t contentLength = bodyStart.length() + imageHeader.length() + len + bodyEnd.length();

  // Agregar la longitud de la solicitud
  request += "Content-Length: " + String(contentLength) + "\r\n";
  request += "Connection: close\r\n\r\n";  // Cerrar la conexión al finalizar

  // Enviar la solicitud HTTP al servidor
  client.print(request);
  client.print(bodyStart);
  client.print(imageHeader);
  client.write(image_buffer, len);  // Enviar la imagen en binario
  client.print(bodyEnd);

  // Leer la respuesta del servidor
  String response = "";
  while (client.connected() || client.available()) {
    if (client.available()) {
      response += client.readStringUntil('\n'); // Leer la respuesta
    }
  }

  client.stop();  // Cerrar conexión
  return response;  // Retornar respuesta del servidor
}


void handleServerResponse(const String& serverResponse) {
  unsigned long currentTime = millis();  // Obtener el tiempo actual

  if (serverResponse.indexOf("\"status\":\"Ojos cerrados\"") >= 0) {
    Serial.println("Ojos cerrados detectados.");

    // Si los ojos están cerrados, actualiza el tiempo del último cierre
    if (!alarmActive) {
      if (lastClosedTime == 0) {
        lastClosedTime = currentTime;
      } else if (currentTime - lastClosedTime >= alarmThreshold) {
        // Si los ojos han estado cerrados por más de 6 segundos, activa la alarma
        alarmActive = true;
        digitalWrite(buzzerPin, HIGH);  // Encender el buzzer
        Serial.println("¡Alarma activada!");
      }
    }
  } else {
    // Si los ojos están abiertos o no se detectan ojos, reinicia el temporizador
    Serial.println("Ojos abiertos o sin detecciones.");
    lastClosedTime = 0;
    alarmActive = false;
    digitalWrite(buzzerPin, LOW);  // Apagar el buzzer
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // Configura el buzzer como salida y apágalo inicialmente
  //pinMode(buzzerPin, OUTPUT);
  //digitalWrite(buzzerPin, LOW);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_SVGA;  
  config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      // Limit the frame size when PSRAM is not available
      config.frame_size = FRAMESIZE_SVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    // Best option for face detection/recognition
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }
  // drop down frame size for higher initial frame rate
  if (config.pixel_format == PIXFORMAT_JPEG) {
    s->set_framesize(s, FRAMESIZE_QVGA);
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

// Setup LED FLash if LED pin is defined in camera_pins.h
#if defined(LED_GPIO_NUM)
  setupLedFlash(LED_GPIO_NUM);
#endif

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  startCameraServer();

  // Endpoint para capturar imágenes de la cámara
  

  // Endpoint para encender el buzzer
 

  // Endpoint para apagar el buzzer
  

  // Inicia el servidor

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");

}

void loop() {
  // Do nothing. Everything is done in another task by the web server
  // Captura una imagen de la cámara
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Error al capturar la imagen");
    return;
  }

   // Enviar la imagen al servidor
  // Enviar la imagen al servidor y obtener la respuesta
  String serverResponse = sendImageToServer(fb->buf, fb->len);

  if (serverResponse != "") {
    Serial.println("Respuesta del servidor recibida:");
    Serial.println(serverResponse);

    // Obtener el tiempo actual
    unsigned long currentTime = millis();

    // Analizar la respuesta del servidor
    if (serverResponse.indexOf("\"status\":\"Ojos cerrados\"") >= 0) {
      Serial.println("Ojos cerrados detectados.");

      // Si los ojos están cerrados, actualiza el tiempo del último cierre
      if (!alarmActive) {
        if (lastClosedTime == 0) {
          lastClosedTime = currentTime;  // Registro inicial del tiempo
        } else if (currentTime - lastClosedTime >= alarmThreshold) {
          // Si los ojos han estado cerrados por más de 6 segundos, activa la alarma
          alarmActive = true;
          digitalWrite(buzzerPin, HIGH);  // Encender el buzzer
          Serial.println("¡Alarma activada!");
        }
      }
    } else {
      // Si los ojos están abiertos o no se detectaron ojos, reinicia el temporizador
      Serial.println("Ojos abiertos o sin detecciones.");
      lastClosedTime = 0;
      alarmActive = false;
      digitalWrite(buzzerPin, LOW);  // Apagar el buzzer
    }
  } else {
    Serial.println("Error al enviar la imagen al servidor.");
  }

  // Liberar la memoria del buffer de la cámara
  esp_camera_fb_return(fb);

  // Esperar antes de la siguiente captura
  delay(5000);  
}