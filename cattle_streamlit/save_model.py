import tensorflow as tf
import json
from tensorflow.keras.optimizers import Adam

with open('cattle_streamlit/cattle_final_archi.json', 'r') as f:
    config = json.load(f)

def fix_config(obj):
    if isinstance(obj, dict):
        obj.pop('groups', None)
        for v in obj.values():
            fix_config(v)
    elif isinstance(obj, list):
        for item in obj:
            fix_config(item)

fix_config(config)

model = tf.keras.Model.from_config(config)
model.load_weights('cattle_streamlit/cattle_final.weights.h5')
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Save as single file
model.save('cattle_streamlit/cattle_final.keras')
print("Done!")