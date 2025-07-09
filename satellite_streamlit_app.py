import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tempfile
import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Title and description
st.markdown('<h1 class="main-header">🛰️ Satellite Image Classification</h1>', unsafe_allow_html=True)
st.markdown("""
This application uses deep learning to classify satellite images into four categories:
**Cloudy**, **Desert**, **Green Area**, and **Water**.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["📊 Dataset Overview", "🔧 Model Training", "🔍 Image Prediction", "📈 Model Analysis"]
)

# Helper functions
def create_improved_model():
    """Create the improved CNN model"""
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fourth Conv Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=16)
    plt.colorbar(im, ax=ax)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12)
    
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    
    plt.tight_layout()
    return fig

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def predict_image(model, img_array, class_names):
    """Predict the class of a single image"""
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    all_predictions = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    
    return predicted_class, confidence, all_predictions

# Page 1: Dataset Overview
if page == "📊 Dataset Overview":
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # File uploader for dataset
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with image paths and labels",
        type=['csv'],
        help="The CSV should have columns: 'image_path' and 'label'"
    )
    
    # Manual dataset creation option
    st.subheader("🛠️ Create Dataset from Folders")
    data_folder = st.text_input(
        "Enter the path to your dataset folder:",
        value="/content/dataset/Satellite Image data",
        help="Path to the folder containing subfolders for each class"
    )
    
    if st.button("Load Dataset from Folders"):
        # Define the labels/classes
        labels = {
            os.path.join(data_folder, "cloudy"): "Cloudy",
            os.path.join(data_folder, "desert"): "Desert",
            os.path.join(data_folder, "green_area"): "Green_Area",
            os.path.join(data_folder, "water"): "Water",
        }
        
        # Create dataset
        data = pd.DataFrame(columns=['image_path', 'label'])
        total_images = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (folder, label) in enumerate(labels.items()):
            status_text.text(f"Processing {label}...")
            
            if os.path.exists(folder):
                image_count = 0
                for image_name in os.listdir(folder):
                    image_path = os.path.join(folder, image_name)
                    if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        new_row = pd.DataFrame({'image_path': [image_path], 'label': [label]})
                        data = pd.concat([data, new_row], ignore_index=True)
                        image_count += 1
                
                st.success(f"{label}: {image_count} images found")
                total_images += image_count
            else:
                st.error(f"Folder {folder} does not exist")
            
            progress_bar.progress((i + 1) / len(labels))
        
        st.session_state.data = data
        status_text.text(f"Dataset created with {total_images} total images")
    
    # Load uploaded CSV
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")
    
    # Display dataset information
    if st.session_state.data is not None:
        st.markdown('<h3 class="section-header">Dataset Information</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Images", len(st.session_state.data))
            st.metric("Number of Classes", st.session_state.data['label'].nunique())
        
        with col2:
            class_counts = st.session_state.data['label'].value_counts()
            st.subheader("Class Distribution")
            st.bar_chart(class_counts)
        
        # Show sample of data
        st.subheader("Sample Data")
        st.dataframe(st.session_state.data.head())
        
        # Class distribution pie chart
        st.subheader("Class Distribution Visualization")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot
        class_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Class Distribution')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Images')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution (Percentage)')
        
        plt.tight_layout()
        st.pyplot(fig)

# Page 2: Model Training
elif page == "🔧 Model Training":
    st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please load a dataset first from the Dataset Overview page.")
    else:
        # Training parameters
        st.subheader("🎛️ Training Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Number of Epochs", 5, 100, 25)
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
            test_size = st.slider("Test Size (fraction)", 0.1, 0.3, 0.2)
        
        with col2:
            image_size = st.selectbox("Image Size", [224, 256, 299], index=0)
            early_stopping_patience = st.slider("Early Stopping Patience", 5, 20, 10)
            learning_rate_patience = st.slider("Learning Rate Patience", 3, 10, 5)
        
        # Data augmentation options
        st.subheader("🔄 Data Augmentation")
        col1, col2 = st.columns(2)
        
        with col1:
            rotation_range = st.slider("Rotation Range", 0, 45, 20)
            width_shift = st.slider("Width Shift Range", 0.0, 0.3, 0.1)
            height_shift = st.slider("Height Shift Range", 0.0, 0.3, 0.1)
        
        with col2:
            zoom_range = st.slider("Zoom Range", 0.0, 0.3, 0.1)
            shear_range = st.slider("Shear Range", 0.0, 0.3, 0.1)
            horizontal_flip = st.checkbox("Horizontal Flip", value=True)
        
        # Start training button
        if st.button("🚀 Start Training"):
            # Prepare data
            df = st.session_state.data
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
            
            st.success(f"Training set: {len(train_df)} images")
            st.success(f"Test set: {len(test_df)} images")
            
            # Create data generators
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=rotation_range,
                width_shift_range=width_shift,
                height_shift_range=height_shift,
                shear_range=shear_range,
                zoom_range=zoom_range,
                horizontal_flip=horizontal_flip,
                fill_mode='nearest'
            )
            
            test_datagen = ImageDataGenerator(rescale=1./255)
            
            # Create generators
            train_generator = train_datagen.flow_from_dataframe(
                dataframe=train_df,
                x_col="image_path",
                y_col="label",
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=True
            )
            
            test_generator = test_datagen.flow_from_dataframe(
                dataframe=test_df,
                x_col="image_path",
                y_col="label",
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False
            )
            
            # Create and compile model
            model = create_improved_model()
            
            # Display model architecture
            st.subheader("🏗️ Model Architecture")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=learning_rate_patience, min_lr=0.0001),
            ]
            
            # Training progress
            st.subheader("📊 Training Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Custom callback for Streamlit progress
            class StreamlitCallback:
                def __init__(self, progress_bar, status_text, total_epochs):
                    self.progress_bar = progress_bar
                    self.status_text = status_text
                    self.total_epochs = total_epochs
                    self.current_epoch = 0
                
                def on_epoch_end(self, epoch, logs=None):
                    self.current_epoch = epoch + 1
                    self.progress_bar.progress(self.current_epoch / self.total_epochs)
                    if logs:
                        self.status_text.text(
                            f"Epoch {self.current_epoch}/{self.total_epochs} - "
                            f"Loss: {logs.get('loss', 0):.4f} - "
                            f"Accuracy: {logs.get('accuracy', 0):.4f} - "
                            f"Val Loss: {logs.get('val_loss', 0):.4f} - "
                            f"Val Accuracy: {logs.get('val_accuracy', 0):.4f}"
                        )
            
            # Train the model
            try:
                history = model.fit(
                    train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    callbacks=callbacks,
                    verbose=0
                )
                
                st.session_state.model = model
                st.session_state.training_history = history
                
                # Evaluate model
                test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
                
                st.success("Training completed successfully!")
                st.metric("Test Accuracy", f"{test_accuracy:.4f}")
                st.metric("Test Loss", f"{test_loss:.4f}")
                
                # Plot training history
                if history:
                    fig = plot_training_history(history)
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

# Page 3: Image Prediction
elif page == "🔍 Image Prediction":
    st.markdown('<h2 class="section-header">Image Prediction</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model first or load a pre-trained model.")
        
        # Option to load pre-trained model
        st.subheader("Load Pre-trained Model")
        uploaded_model = st.file_uploader("Upload a trained model (.h5 file)", type=['h5'])
        
        if uploaded_model is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                    tmp_file.write(uploaded_model.read())
                    tmp_file_path = tmp_file.name
                
                # Load model
                st.session_state.model = load_model(tmp_file_path)
                st.success("Model loaded successfully!")
                
                # Clean up
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    else:
        st.subheader("🖼️ Upload Image for Prediction")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a satellite image for classification"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image_pil = Image.open(uploaded_image)
                st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Prediction Results")
                
                # Preprocess image
                img_array = np.array(image_pil.resize((224, 224)))
                
                # Make prediction
                predicted_class, confidence, all_predictions = predict_image(
                    st.session_state.model, img_array, st.session_state.class_names
                )
                
                # Display results
                st.metric("Predicted Class", predicted_class)
                st.metric("Confidence", f"{confidence:.2%}")
                
                # Show all predictions
                st.subheader("All Class Probabilities")
                for class_name, prob in all_predictions.items():
                    st.progress(prob)
                    st.write(f"{class_name}: {prob:.2%}")
                
                # Confidence indicator
                if confidence > 0.8:
                    st.success("High confidence prediction")
                elif confidence > 0.6:
                    st.warning("Medium confidence prediction")
                else:
                    st.error("Low confidence prediction")

# Page 4: Model Analysis
elif page == "📈 Model Analysis":
    st.markdown('<h2 class="section-header">Model Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None or st.session_state.training_history is None:
        st.warning("Please train a model first to see analysis results.")
    else:
        # Training history plots
        st.subheader("📊 Training History")
        fig = plot_training_history(st.session_state.training_history)
        st.pyplot(fig)
        
        # Model performance metrics
        st.subheader("🎯 Model Performance")
        
        if st.session_state.data is not None:
            # Recreate test data for analysis
            df = st.session_state.data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
            
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_dataframe(
                dataframe=test_df,
                x_col="image_path",
                y_col="label",
                target_size=(224, 224),
                batch_size=32,
                class_mode="categorical",
                shuffle=False
            )
            
            # Generate predictions
            predictions = st.session_state.model.predict(test_generator)
            actual_labels = test_generator.classes
            predicted_labels = np.argmax(predictions, axis=1)
            
            # Confusion matrix
            cm = confusion_matrix(actual_labels, predicted_labels)
            fig_cm = plot_confusion_matrix(cm, classes=st.session_state.class_names)
            st.pyplot(fig_cm)
            
            # Classification report
            st.subheader("📋 Classification Report")
            report = classification_report(
                actual_labels, predicted_labels, 
                target_names=st.session_state.class_names,
                output_dict=True
            )
            
            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Per-class metrics
            st.subheader("📊 Per-Class Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                precision_data = {name: report[name]['precision'] for name in st.session_state.class_names}
                st.bar_chart(precision_data)
                st.caption("Precision by Class")
            
            with col2:
                recall_data = {name: report[name]['recall'] for name in st.session_state.class_names}
                st.bar_chart(recall_data)
                st.caption("Recall by Class")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🛰️ Satellite Image Classification App</p>
    <p>Built with Streamlit and TensorFlow</p>
</div>
""", unsafe_allow_html=True)
