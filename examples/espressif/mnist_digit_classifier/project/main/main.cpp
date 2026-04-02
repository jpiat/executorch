/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * MNIST Handwritten Digit Classifier for ESP32/ESP32-S3.
 *
 * Uses the ExecuTorch espressif executor runner (et_runner_* API) to run
 * a TinyMLPMNIST model that classifies handwritten digits 0, 1, 4, and 7.
 *
 * Based on the Raspberry Pi Pico 2 MNIST example.
 */

#include <cstring>
#include <stdio.h>

#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"

#include "esp_executor_runner.h"

static const int IMAGE_SIZE = 28;
static const int NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE; // 784
static const int NUM_CLASSES = 10;

// ASCII art digit patterns (28x28 characters each).
// '#' represents a white pixel (1.0), space represents black (0.0).

static const char* ascii_digit_0[28] = {
    "                            ",
    "        ############        ",
    "      ##################    ",
    "    ######################  ",
    "   ######################## ",
    "  ####                ####  ",
    " ####                  #### ",
    " ####                  #### ",
    "####                    ####",
    "####                    ####",
    "####                    ####",
    "####                    ####",
    "####                    ####",
    "####                    ####",
    "####                    ####",
    "####                    ####",
    "####                    ####",
    "####                    ####",
    "####                    ####",
    "####                    ####",
    " ####                  #### ",
    " ####                  #### ",
    "  ####                ####  ",
    "   ######################## ",
    "    ######################  ",
    "      ##################    ",
    "        ############        ",
    "                            "};

static const char* ascii_digit_1[28] = {
    "            ####            ",
    "           #####            ",
    "          ######            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "            ####            ",
    "        ############        ",
    "        ############        ",
    "        ############        ",
    "                            "};

static const char* ascii_digit_4[28] = {
    "                            ",
    "               ####         ",
    "              #####         ",
    "             ######         ",
    "            #######         ",
    "           #### ####        ",
    "          ####  ####        ",
    "         ####   ####        ",
    "        ####    ####        ",
    "       ####     ####        ",
    "      ####      ####        ",
    "     ####       ####        ",
    "    ####        ####        ",
    "   ####         ####        ",
    "  ######################    ",
    "  ######################    ",
    "  ######################    ",
    "                ####        ",
    "                ####        ",
    "                ####        ",
    "                ####        ",
    "                ####        ",
    "                ####        ",
    "                ####        ",
    "                ####        ",
    "                ####        ",
    "                ####        ",
    "                            "};

static const char* ascii_digit_7[28] = {
    "############################",
    "############################",
    "                        ####",
    "                       #### ",
    "                      ####  ",
    "                     ####   ",
    "                    ####    ",
    "                   ####     ",
    "                  ####      ",
    "                 ####       ",
    "                ####        ",
    "               ####         ",
    "              ####          ",
    "             ####           ",
    "            ####            ",
    "           ####             ",
    "          ####              ",
    "         ####               ",
    "        ####                ",
    "       ####                 ",
    "      ####                  ",
    "     ####                   ",
    "    ####                    ",
    "   ####                     ",
    "  ####                      ",
    " ####                       ",
    "####                        ",
    "###                         "};

struct TestCase {
    const char** pattern;
    const char* name;
    int expected_digit;
};

static TestCase test_cases[] = {
    {ascii_digit_0, "Digit 0", 0},
    {ascii_digit_1, "Digit 1", 1},
    {ascii_digit_4, "Digit 4", 4},
    {ascii_digit_7, "Digit 7", 7}};

static const int NUM_TESTS = sizeof(test_cases) / sizeof(test_cases[0]);

static void ascii_to_float(const char** ascii_digit, float* output) {
    for (int row = 0; row < IMAGE_SIZE; row++) {
        for (int col = 0; col < IMAGE_SIZE; col++) {
            output[row * IMAGE_SIZE + col] =
                (ascii_digit[row][col] == '#') ? 1.0f : 0.0f;
        }
    }
}

static int find_predicted_digit(const float* scores, int num_classes) {
    int predicted = 0;
    float max_score = scores[0];
    for (int i = 1; i < num_classes; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
            predicted = i;
        }
    }
    return predicted;
}

static bool run_mnist_tests(void) {
    printf("ExecuTorch MLP MNIST Demo on ESP32\n");
    printf("Testing all supported digits:\n\n");

    int correct = 0;

    // Input tensor shape is [1, 28, 28] = 784 floats
    float input_data[NUM_PIXELS];
    float output_data[NUM_CLASSES];

    for (int test = 0; test < NUM_TESTS; test++) {
        const char** ascii_digit = test_cases[test].pattern;
        const char* digit_name = test_cases[test].name;
        int expected = test_cases[test].expected_digit;

        printf("=== %s ===\n", digit_name);
        for (int i = 0; i < IMAGE_SIZE; i++) {
            printf("%s\n", ascii_digit[i]);
        }
        printf("\n");

        ascii_to_float(ascii_digit, input_data);

        int white_pixels = 0;
        for (int i = 0; i < NUM_PIXELS; i++) {
            if (input_data[i] > 0.5f)
                white_pixels++;
        }
        printf("Input stats: %d white pixels out of %d total\n",
               white_pixels, NUM_PIXELS);

        // Set input and run inference using the et_runner API
        if (!et_runner_set_input(0, input_data, sizeof(input_data))) {
            printf("Failed to set input!\n");
            return false;
        }

        printf("Running inference...\n");
        if (!et_runner_execute()) {
            printf("Failed to execute inference!\n");
            return false;
        }

        size_t num_elements = 0;
        if (!et_runner_get_output(
                0, output_data, sizeof(output_data), &num_elements)) {
            printf("Failed to get output!\n");
            return false;
        }

        int predicted_digit = find_predicted_digit(output_data, NUM_CLASSES);

        printf("Neural network results:\n");
        for (int i = 0; i < NUM_CLASSES; i++) {
            printf("  Digit %d: %.3f", i, output_data[i]);
            if (i == predicted_digit)
                printf(" <- PREDICTED");
            printf("\n");
        }

        printf("\nPREDICTED: %d (Expected: %d) ", predicted_digit, expected);
        if (predicted_digit == expected) {
            printf("CORRECT!\n");
            correct++;
        } else {
            printf("WRONG!\n");
        }

        printf("\n==================================================\n\n");
    }

    printf("Results: %d/%d correct\n", correct, NUM_TESTS);
    return correct == NUM_TESTS;
}

extern "C" void app_main(void) {
    printf("Starting MNIST digit classifier!\n");
    fflush(stdout);

    if (!et_runner_init()) {
        printf("Failed to initialize ExecuTorch runner!\n");
        return;
    }
    printf("ExecuTorch runner initialized.\n");

    bool all_correct = run_mnist_tests();

    if (all_correct) {
        printf("All tests passed! ExecuTorch MNIST inference works on ESP32!\n");
    } else {
        printf("Some tests failed.\n");
    }

    for (int i = 5; i >= 0; i--) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    esp_restart();
}
