/*
  RakshPulse MAX30102 streaming firmware (reference implementation for ESP32 / Arduino)
  ===================================================================================

  Streams raw red + infrared MAX30102 samples over USB serial in the format the RakshPulse software expects:

      # rakshpulse fw=1 fs_hz=100 sample_rate=100 averaging=1 pulse_width_us=411 adc_range_na=16384 led_red=0x24 led_ir=0x24
      0,95321,118204
      1,95318,118199
      ...

  * Data lines: "sample_index,red,ir" (decimal integers, '\n' terminated). Nothing else is printed between data
    lines. Status / error messages start with '#', and the software skips them as comments.
  * sample_index is a 32-bit counter that increases by 1 per sample. If the sensor FIFO overflowed (samples lost
    because the loop was too slow), the counter is advanced by the number of lost samples, so the software logs a
    gap instead of silently stitching time.
  * fs_hz in the banner is the EFFECTIVE output rate = SAMPLE_RATE / SAMPLE_AVERAGE. Pass the same value to
    the software: rakshpulse record --source serial --fs <fs_hz> ...

  Library: "SparkFun MAX3010x Pulse and Proximity Sensor Library" (Arduino Library Manager; header MAX30105.h, it
  supports the MAX30102).

  HARDWARE-SPECIFIC VALUES THE HARDWARE TEAM MUST CONFIRM:
  * I2C pins: Wire.begin() uses the board's default SDA/SCL. On ESP32 boards whose default pins differ, set
    I2C_SDA_PIN / I2C_SCL_PIN below to the pins actually wired. No pin assignment is assumed here.
  * LED_POWER (LED drive current) depends on the skin site, ambient light and enclosure: adjust it so the raw IR value
    sits well inside the 18-bit range (about 30-70 % of 262143) without clipping at 262143.
  * Supply voltage and wiring follow the breakout board's documentation.

  This sketch has NOT been compiled or run on hardware by the software team (no board or toolchain was available).
*/

#include <Wire.h>
#include "MAX30105.h"

// ----------------------------------------------------------------- configuration (keep in sync with --fs)
#define SERIAL_BAUD     115200
#define SAMPLE_RATE     100     // sensor sample rate: 50, 100, 200, 400, 800, 1000, 1600, 3200
#define SAMPLE_AVERAGE  1       // FIFO averaging: 1, 2, 4, 8, 16, 32 -> effective rate = SAMPLE_RATE / SAMPLE_AVERAGE
#define PULSE_WIDTH     411     // 69, 118, 215, 411 us (411 us = 18-bit ADC)
#define ADC_RANGE       16384   // 2048, 4096, 8192, 16384 nA
#define LED_POWER       0x24    // LED current register value (0x00-0xFF, ~0.2 mA per step); tune per site
#define LED_MODE        2       // 2 = red + IR (SpO2 mode)
#define I2C_SDA_PIN     -1      // -1 = board default; set explicitly on ESP32 if needed
#define I2C_SCL_PIN     -1
#define MAX30102_ADDR   0x57
#define REG_OVF_COUNTER 0x05

MAX30105 sensor;
uint32_t sampleIndex = 0;

static uint8_t readRegister(uint8_t reg) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0;
  Wire.requestFrom((uint8_t)MAX30102_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) { delay(10); }  // native-USB boards wait for the port; harmless elsewhere

#if defined(ESP32)
  if (I2C_SDA_PIN >= 0 && I2C_SCL_PIN >= 0) {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  } else {
    Wire.begin();
  }
#else
  Wire.begin();
#endif

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    // Never start streaming without a sensor: report and stop. The software treats '#' lines as comments and will
    // report "no valid samples" with troubleshooting hints.
    while (true) {
      Serial.println("# ERROR: MAX30102 not found on I2C (check wiring, power, SDA/SCL pins)");
      delay(2000);
    }
  }
  sensor.setup(LED_POWER, SAMPLE_AVERAGE, LED_MODE, SAMPLE_RATE, PULSE_WIDTH, ADC_RANGE);
  sensor.clearFIFO();

  Serial.print("# rakshpulse fw=1 fs_hz=");
  Serial.print(SAMPLE_RATE / SAMPLE_AVERAGE);
  Serial.print(" sample_rate=");
  Serial.print(SAMPLE_RATE);
  Serial.print(" averaging=");
  Serial.print(SAMPLE_AVERAGE);
  Serial.print(" pulse_width_us=");
  Serial.print(PULSE_WIDTH);
  Serial.print(" adc_range_na=");
  Serial.print(ADC_RANGE);
  Serial.print(" led_red=0x");
  Serial.print(LED_POWER, HEX);
  Serial.print(" led_ir=0x");
  Serial.println(LED_POWER, HEX);
}

void loop() {
  // Samples lost while the FIFO was full: advance the counter so the gap is visible downstream.
  uint8_t lost = readRegister(REG_OVF_COUNTER) & 0x1F;
  if (lost > 0) {
    sampleIndex += lost;
  }
  sensor.check();  // reads all new FIFO samples into the library's buffer
  while (sensor.available()) {
    uint32_t red = sensor.getFIFORed();
    uint32_t ir = sensor.getFIFOIR();
    Serial.print(sampleIndex);
    Serial.print(',');
    Serial.print(red);
    Serial.print(',');
    Serial.println(ir);
    sampleIndex++;
    sensor.nextSample();
  }
}
