#include "model_inference.h"

water_state_t water_level_predict(const water_level_features_t *f)
{
    if (f == 0) {
        return WATER_STATE_STABLE;
    }

    if (f->recent_max_30m <= 6.775000f) {
        if (f->delta_30m <= -0.185000f) {
            if (f->max_5m <= 6.520000f) {
                return WATER_STATE_STABLE;
            }
            return WATER_STATE_RECEDING;
        }

        if (f->dist_to_safe_return <= 0.005000f) {
            return WATER_STATE_STABLE;
        }
        return WATER_STATE_STABLE;
    }

    if (f->drawdown_from_recent_max_30m <= 0.100993f) {
        if (f->water_level <= 8.008458f) {
            return WATER_STATE_RISING;
        }
        return WATER_STATE_FLOOD;
    }

    if (f->slope_10m <= 0.000326f) {
        return WATER_STATE_RECEDING;
    }
    return WATER_STATE_STABLE;
}

const char *water_state_name(water_state_t state)
{
    switch (state) {
    case WATER_STATE_STABLE:
        return "Stable";
    case WATER_STATE_RISING:
        return "Rising";
    case WATER_STATE_FLOOD:
        return "Flood";
    case WATER_STATE_RECEDING:
        return "Receding";
    default:
        return "Unknown";
    }
}
