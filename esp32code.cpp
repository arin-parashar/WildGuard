#include <Adafruit_NeoPixel.h>

#define LED1_PIN 5
#define LED2_PIN 21
#define LED3_PIN 22
#define BUZZER_PIN 25   // Active LOW relay (LOW = ON)

#define NUM_LEDS 1

Adafruit_NeoPixel led1(NUM_LEDS, LED1_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel led2(NUM_LEDS, LED2_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel led3(NUM_LEDS, LED3_PIN, NEO_GRB + NEO_KHZ800);

String cmd = "";

// ===== STATES =====
bool dangerMode = false;
bool triggerMode = false;
bool buzzerMode = false;

unsigned long lastBlink2 = 0;
unsigned long lastBlink3 = 0;
unsigned long lastBuzz = 0;

int buzzerInterval = 0;

bool led2State = false;
bool led3State = false;
int led2ColorIndex = 0;

void setup() {
  Serial.begin(115200);

  led1.begin(); led1.clear(); led1.show();
  led2.begin(); led2.clear(); led2.show();
  led3.begin(); led3.clear(); led3.show();

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, HIGH);   // Relay OFF at startup
}

void loop() {

  // ===== SERIAL READ =====
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      cmd.trim();
      cmd.toLowerCase();
      handleCommand(cmd);
      cmd = "";
    } else {
      cmd += c;
    }
  }

  unsigned long now = millis();

  // ===== LED2 Rapid Danger Flash =====
  if (dangerMode) {
    if (now - lastBlink2 >= 100) {
      lastBlink2 = now;
      led2State = !led2State;
      led2ColorIndex++;
      if (led2ColorIndex > 5) led2ColorIndex = 0;

      if (led2State) {
        switch (led2ColorIndex) {
          case 0: led2.setPixelColor(0, led2.Color(255,0,0)); break;
          case 1: led2.setPixelColor(0, led2.Color(0,255,0)); break;
          case 2: led2.setPixelColor(0, led2.Color(0,0,255)); break;
          case 3: led2.setPixelColor(0, led2.Color(255,255,0)); break;
          case 4: led2.setPixelColor(0, led2.Color(0,255,255)); break;
          case 5: led2.setPixelColor(0, led2.Color(255,0,255)); break;
        }
      } else {
        led2.clear();
      }
      led2.show();
    }
  }

  // ===== LED3 Orange Blink During Trigger =====
  if (triggerMode) {
    if (now - lastBlink3 >= 400) {
      lastBlink3 = now;
      led3State = !led3State;

      if (led3State)
        led3.setPixelColor(0, led3.Color(255,100,0)); // ORANGE
      else
        led3.clear();

      led3.show();
    }
  }

  // ===== BUZZER CONTROL (Active LOW Relay) =====
  if (buzzerMode && buzzerInterval > 0) {
    if (now - lastBuzz >= buzzerInterval) {
      lastBuzz = now;
      digitalWrite(BUZZER_PIN, LOW);   // ON
      delay(50);
      digitalWrite(BUZZER_PIN, HIGH);  // OFF
    }
  }
}

void handleCommand(String c) {

  // ===== IDLE (App Launch) =====
  if (c == "idle") {
    led1.setPixelColor(0, led1.Color(255,255,0)); // Yellow
    led1.show();
  }

  // ===== WEBCAM MODE =====
  else if (c == "start") {
    led1.setPixelColor(0, led1.Color(0,255,0)); // Green
    led1.show();
  }

  // ===== FILE MODE =====
  else if (c == "camera on") {
    led1.setPixelColor(0, led1.Color(255,0,0)); // Red
    led1.show();
  }

  // ===== TRIGGER =====
  else if (c == "trigger") {
    triggerMode = true;
  }

  // ===== SAFE =====
  else if (c == "safe") {
    triggerMode = false;
    dangerMode = false;
    buzzerMode = false;
    digitalWrite(BUZZER_PIN, HIGH);

    led2.clear(); led2.show();
    led3.setPixelColor(0, led3.Color(0,255,0));
    led3.show();
  }

  // ===== NORMAL =====
  else if (c == "normal") {
    triggerMode = false;
    dangerMode = false;

    led3.setPixelColor(0, led3.Color(255,255,0));
    led3.show();

    buzzerMode = true;
    buzzerInterval = 3000;  // 3 sec
  }

  // ===== DANGER =====
  else if (c == "danger") {
    triggerMode = false;
    dangerMode = true;

    led3.setPixelColor(0, led3.Color(255,0,0));
    led3.show();

    buzzerMode = true;
    buzzerInterval = 500;   // 0.5 sec
  }

  // ===== SYSTEM OFF =====
  else if (c == "system off") {

    triggerMode = false;
    dangerMode = false;
    buzzerMode = false;

    led1.clear(); led1.show();
    led2.clear(); led2.show();
    led3.clear(); led3.show();

    digitalWrite(BUZZER_PIN, HIGH); // Relay OFF
  }
}