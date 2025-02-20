import numpy as np
import pandas as pd
import logging
import logging.handlers
from logging.handlers import RotatingFileHandler
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
import torch
import joblib
from datetime import datetime

# Configure logging
logger = logging.getLogger("ModelTraining")
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler("model_training.log", maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Makes sure our data is clean and ready for training
    
    Checks for required columns and handles any missing or invalid values
    """
    required_columns = [
        "Date", "Open", "High", "Low", "Close", "Volume", 
        "Daily Return", "Volatility", "Price_Range", "Price_Range_Pct",
        "Volume_Change", "Log Return", "Volatility_7d", "Volatility_7d_Ann",
        "Volatility_14d", "Volatility_14d_Ann", "Volatility_30d", "Volatility_30d_Ann"
    ]
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Handle NaN values with forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Log data statistics
    logger.info(f"Data shape after cleaning: {df.shape}")
    
    return df

class HybridFraudDetector:
    def __init__(self, transformer_model="distilbert-base-uncased"):
        """Sets up our fraud detection system using both ML and outlier detection
        
        Uses a transformer for text analysis and traditional ML for numerical data
        """
        logger.info("Initializing HybridFraudDetector")
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.transformer = AutoModel.from_pretrained(transformer_model)
        self.rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
        self.scaler = RobustScaler()
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Converts text into a numerical representation for analysis
        
        Returns zeros if the text can't be processed
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            outputs = self.transformer(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().flatten()
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text}': {e}")
            return np.zeros(768)
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Gets our data ready for model training or prediction
        
        Takes raw stock data and extracts the important metrics we need
        Logs what features we're using to help with debugging
        """
        logger.info("Preparing features for model")
        
        feature_columns = [
            "Daily Return",
            "Volatility",
            "Price_Range",
            "Price_Range_Pct",
            "Volume_Change",
            "Log Return",
            "Volatility_7d",
            "Volatility_7d_Ann",
            "Volatility_14d",
            "Volatility_14d_Ann",
            "Volatility_30d",
            "Volatility_30d_Ann"
        ]
        
        # Verify all required columns exist
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract features
        X = df[feature_columns].values
        
        # Replace any remaining infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Log feature statistics
        logger.info(f"Feature array shape: {X.shape}")
        logger.info(f"Number of features used: {len(feature_columns)}")
        logger.info("Features included: " + ", ".join(feature_columns))
        logger.info(f"Feature stats before scaling:\n{pd.DataFrame(X, columns=feature_columns).describe()}")
        
        return X

    def train(self, df: pd.DataFrame):
        """Trains our hybrid detection system
        
        Combines outlier detection with traditional classification
        to catch both known and unknown patterns
        """
        logger.info("Starting training process")
        
        try:
            # Clean and validate data
            df = clean_and_validate_data(df)
            
            # Prepare features
            X = self.prepare_features(df)
            
            # Scale the features
            X = self.scaler.fit_transform(X)
            
            # Split data into train and validation sets
            X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
            
            # Fit LOF model
            logger.info("Training LOF model")
            self.lof.fit(X_train)
            
            # Get LOF scores for both sets
            lof_train = -self.lof.decision_function(X_train).reshape(-1, 1)
            lof_val = -self.lof.decision_function(X_val).reshape(-1, 1)
            
            # Generate labels based on LOF scores (90th percentile threshold)
            threshold = np.percentile(lof_train, 90)
            y_train = (lof_train.flatten() > threshold).astype(int)
            y_val = (lof_val.flatten() > threshold).astype(int)
            
            # Combine features with LOF scores
            X_train_aug = np.hstack([X_train, lof_train])
            X_val_aug = np.hstack([X_val, lof_val])
            
            # Train Random Forest
            logger.info("Training Random Forest model")
            self.rf.fit(X_train_aug, y_train)
            
            # Evaluate model
            y_pred = self.rf.predict(X_val_aug)
            report = classification_report(y_val, y_pred)
            logger.info(f"Model Performance:\n{report}")
            
            # Save models
            self.save_models()
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def save_models(self):
        """Saves our trained models
        
        Creates both timestamped and default versions for backup
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save with timestamp
        joblib.dump(self.rf, f"rf_model_{timestamp}.pkl")
        joblib.dump(self.lof, f"lof_model_{timestamp}.pkl")
        joblib.dump(self.scaler, f"scaler_{timestamp}.pkl")
        
        # Save as default names
        joblib.dump(self.rf, "rf_model.pkl")
        joblib.dump(self.lof, "lof_model.pkl")
        joblib.dump(self.scaler, "scaler.pkl")
        
        logger.info("Models saved successfully")

    def predict(self, df: pd.DataFrame) -> tuple:
        """Makes predictions on new stock data
        
        Returns:
            Two arrays: binary predictions and risk probabilities
        """
        try:
            X = self.prepare_features(df)
            X = self.scaler.transform(X)
            lof_scores = -self.lof.decision_function(X).reshape(-1, 1)
            X_aug = np.hstack([X, lof_scores])
            predictions = self.rf.predict(X_aug)
            probabilities = self.rf.predict_proba(X_aug)[:, 1]
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Load data
        logger.info("Loading data from CSV")
        df = pd.read_csv("multi_stock_data.csv", parse_dates=["Date"])
        
        # Initialize and train model
        detector = HybridFraudDetector()
        detector.train(df)
        
        # Test predictions on sample data
        sample_df = df.head(100)
        predictions, probabilities = detector.predict(sample_df)
        
        logger.info(f"Sample predictions shape: {predictions.shape}")
        logger.info(f"Prediction distribution: {np.bincount(predictions)}")
        logger.info("Model training and testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")