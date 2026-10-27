# Age Group Prediction App

A small deep learning project that guesses a person's age group from a photo of their face. A convolutional neural network (CNN) is trained on the UTKFace dataset in a Jupyter notebook, then served through a Streamlit web app where you can upload an image or take a photo with your camera.

## Age groups

The model sorts each face into one of six classes:

| Class | Age range |
|-------|-----------|
| 0 | 0–12 |
| 1 | 13–19 |
| 2 | 20–29 |
| 3 | 30–44 |
| 4 | 45–59 |
| 5 | 60+ |

## Project structure

```
age_prediction_app/
├── DL_LAP2.ipynb      # data loading, cleaning, training and evaluation
├── age_model.keras    # trained model used by the app
├── app.py             # Streamlit app
├── requirements.txt   # Python packages for the app
└── README.md
```

## Dataset

[UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) contains about 23,700 face images, cropped and aligned to 200×200 pixels. Each file name holds the labels in the form `age_gender_race_datetime.jpg`, and the notebook reads the age from it.

Preparation steps in the notebook:

1. Download the dataset with `kagglehub`.
2. Parse age, gender and race from the file names (files without all four parts are skipped).
3. Bin ages into the six groups above.
4. Find exact duplicate images by MD5 hash. Duplicates whose copies have different ages are dropped entirely; the rest are kept once.
5. Resize every image to 64×64 RGB and scale pixel values to 0–1.
6. Split into train / validation / test sets (64% / 16% / 20%), stratified by age group.

The classes are imbalanced (20–29 is the largest group and 13–19 the smallest), so the model is trained with balanced class weights.

## Model

A simple Keras `Sequential` CNN:

```
Input (64, 64, 3)
Conv2D(30, 3×3, relu) → MaxPool2D(2×2)
Conv2D(40, 3×3, relu)
Flatten
Dense(100, relu) → Dropout(0.5)
Dense(6, softmax)
```

- Loss: sparse categorical cross-entropy
- Optimizer: Adam
- Up to 20 epochs, batch size 32
- Early stopping on validation loss (patience 3, best weights restored)

The notebook evaluates the model on the held-out test set and plots a confusion matrix. Accuracy is roughly 50%. Most mistakes land in a neighbouring age group, which is expected since the boundaries between groups are arbitrary and faces near a boundary look alike.

## Running the app

Requires Python 3.10 or newer.

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. install the dependencies
pip install -r requirements.txt

# 3. start the app
streamlit run app.py
```

Streamlit opens the app in your browser (usually at http://localhost:8501).

### Using the app

1. Pick the **Upload** tab and choose a `.jpg`, `.jpeg` or `.png` face image, or pick **Take a photo** and use your webcam.
2. The app resizes the image to 64×64, runs the model and shows:
   - the predicted age group and its confidence
   - a bar chart of the probability for every age group

If you provide both an upload and a camera photo, the camera photo is used.

## Retraining the model

Open `DL_LAP2.ipynb` and run all cells. The notebook downloads the dataset (about 330 MB) on first run, trains the model and saves it as `age_model.keras`. Copy the new file next to `app.py` to use it in the app.

Training needs these extra packages on top of `requirements.txt`:

```bash
pip install kagglehub pandas matplotlib seaborn scikit-learn
```

## Limitations

- About 50% accuracy, so treat each prediction as a rough guess.
- Works best on close-up, front-facing faces similar to UTKFace images. The app does not detect or crop faces, so full-body photos or busy backgrounds will hurt the result.
- Images are shrunk to 64×64, which throws away fine detail such as wrinkles.
- UTKFace is not evenly balanced across ages and ethnicities, so accuracy may differ between groups of people.
