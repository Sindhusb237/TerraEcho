def predict_soil(features):

    freq = features["frequency"]
    energy = features["energy"]

    soil_type = ""
    moisture = ""
    compaction = ""
    dryness = ""

    if freq > 3200:

        soil_type = "Dry Soil"
        moisture = "Low"
        compaction = "Low"
        dryness = "High"

    elif freq > 2000:

        soil_type = "Healthy Soil"
        moisture = "Moderate"
        compaction = "Medium"
        dryness = "Low"

    else:

        soil_type = "Compact Soil"
        moisture = "High"
        compaction = "High"
        dryness = "Low"

    health_score = min(
        100,
        int((energy * 1000) + 65)
    )

    if soil_type == "Dry Soil":
        recommendation = "Irrigation Needed"

    elif soil_type == "Compact Soil":
        recommendation = "Soil Aeration Recommended"

    else:
        recommendation = "Soil Condition Good"

    return {
        "soil_type": soil_type,
        "moisture": moisture,
        "compaction": compaction,
        "dryness": dryness,
        "health_score": health_score,
        "recommendation": recommendation
    }