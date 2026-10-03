#ifndef WATER_LEVEL_MODEL_INFERENCE_H
#define WATER_LEVEL_MODEL_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WATER_STATE_STABLE = 0,
    WATER_STATE_RISING = 1,
    WATER_STATE_FLOOD = 2,
    WATER_STATE_RECEDING = 3
} water_state_t;

typedef struct {
    float water_level;
    float max_5m;
    float delta_30m;
    float recent_max_30m;
    float drawdown_from_recent_max_30m;
    float slope_10m;
    float dist_to_safe_return;
} water_level_features_t;

water_state_t water_level_predict(const water_level_features_t *features);
const char *water_state_name(water_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* WATER_LEVEL_MODEL_INFERENCE_H */
