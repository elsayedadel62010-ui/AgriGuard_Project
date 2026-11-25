#include "wokwi-api.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h> // دي المكتبة اللي هتحل كل مشاكل كلمة false

typedef struct {
  pin_t pin_out;
  timer_t timer;
  uint32_t flow_attr;
  bool state;
} chip_state_t;

static void chip_timer_callback(void *user_data) {
  chip_state_t *chip = (chip_state_t *)user_data;
  uint32_t flow_rate = attr_read(chip->flow_attr); // Flow rate in L/min

  if (flow_rate > 0) {
    // Frequency (Hz) = 7.5 * Flow Rate (L/min) for YF-S201
    float freq = 7.5f * flow_rate;
    uint32_t period_us = (uint32_t)(1000000.0f / (freq * 2.0f));

    chip->state = !chip->state;
    pin_write(chip->pin_out, chip->state ? HIGH : LOW);
    timer_start(chip->timer, period_us, false);
  } else {
    pin_write(chip->pin_out, LOW);
    timer_start(chip->timer, 100000, false); // 100ms in microseconds
  }
}

void chip_init(void) {
  chip_state_t *chip = malloc(sizeof(chip_state_t));
  chip->pin_out = pin_init("OUT", OUTPUT);
  chip->flow_attr = attr_init("flowRate", 0);
  chip->state = false;

  const timer_config_t timer_config = {
    .callback = chip_timer_callback,
    .user_data = chip,
  };
  chip->timer = timer_init(&timer_config);

  // Start initial timer checking
  timer_start(chip->timer, 100000, false);
}