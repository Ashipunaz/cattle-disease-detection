import json
import tensorflow as tf

ARCH_PATH = r"D:\cattle-disease-detection\cattle_final_archi.json"
WEIGHTS_PATH = r"D:\cattle-disease-detection\cattle_final.weights.h5"
OUTPUT_H5 = r"D:\cattle-disease-detection\cattle_disease_model.h5"
OUTPUT_SAVEDMODEL = r"D:\cattle-disease-detection\cattle_disease_model"

def main():
    # 1. Load architecture config from JSON
    with open(ARCH_PATH, "r") as f:
        config = json.load(f)

    # 2. Rebuild model from config
    model = tf.keras.Model.from_config(config)

    # 3. Load trained weights
    model.load_weights(WEIGHTS_PATH)
    print("Model loaded. Total params:", model.count_params())

    # 4. Save complete model in both formats
    print("Saving H5 model to:", OUTPUT_H5)
    model.save(OUTPUT_H5)
    print("Done.")

if __name__ == "__main__":
    main()