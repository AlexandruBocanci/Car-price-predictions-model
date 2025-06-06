import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class CarPricePredictor:
    """
    Clasa pentru predicția prețurilor mașinilor second-hand
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_data(self, file_path):
        """
        Încarcă datele din fișierul CSV
        """
        print("Încarcarea datelor...")
        self.df = pd.read_csv(file_path)
        print(f"Date încărcate: {self.df.shape[0]} rânduri, {self.df.shape[1]} coloane")
        return self.df
    
    def clean_data(self):
        """
        Curăță datele conform cerințelor
        """
        print("Curățarea datelor...")
        
        # Coloanele de eliminat
        columns_to_drop = [
            'id', 'url', 'region', 'region_url', 'VIN', 
            'image_url', 'description', 'county', 'state', 
            'lat', 'long', 'posting_date'
        ]
        
        # Elimină coloanele specificate
        existing_columns_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=existing_columns_to_drop)
        print(f"Eliminate {len(existing_columns_to_drop)} coloane")
        
        # Numărul de rânduri înainte de curățare
        initial_rows = len(self.df)
        
        # Elimină rândurile cu valori lipsă
        self.df = self.df.dropna()
        print(f"Eliminate {initial_rows - len(self.df)} rânduri cu valori lipsă")
        
        # Convertește tipurile de date pentru coloanele numerice
        numeric_columns = ['price', 'year', 'odometer']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Elimină rândurile cu valori invalide după conversie
        self.df = self.df.dropna()
        
        # Filtrează valorile extreme
        self.df = self.df[
            (self.df['price'] > 500) & (self.df['price'] < 100000) &
            (self.df['year'] >= 1990) & (self.df['year'] <= 2025) &
            (self.df['odometer'] >= 0) & (self.df['odometer'] <= 500000)
        ]
        
        print(f"Date finale după curățare: {self.df.shape[0]} rânduri")
        return self.df
    
    def prepare_features(self):
        """
        Pregătește caracteristicile pentru antrenament
        """
        print("Pregătirea caracteristicilor...")
        
        # Identifică coloanele categorice și numerice
        self.numerical_columns = ['year', 'odometer']
        self.categorical_columns = [col for col in self.df.columns 
                                  if col not in self.numerical_columns + ['price']]
        
        # Aplică Label Encoding pentru coloanele categorice
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        # Definește caracteristicile și ținta
        self.feature_columns = self.numerical_columns + self.categorical_columns
        X = self.df[self.feature_columns]
        y = self.df['price']
        
        print(f"Caracteristici pregătite: {len(self.feature_columns)} coloane")
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Antrenează modelul de regresie
        """
        print("Antrenarea modelului...")
        
        # Împarte datele în set de antrenare și test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Antrenează modelul Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluează modelul
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"Rezultatele evaluării:")
        print(f"RMSE Antrenare: ${train_rmse:.2f}")
        print(f"RMSE Test: ${test_rmse:.2f}")
        print(f"MAE Antrenare: ${train_mae:.2f}")
        print(f"MAE Test: ${test_mae:.2f}")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': test_pred
        }
    
    def predict_price(self, manufacturer, model, year, odometer, fuel=None, 
                     condition=None, transmission=None, vehicle_type=None, 
                     cylinders=None, title_status=None, drive=None, 
                     size=None, paint_color=None):
        """
        Prezice prețul unei mașini pe baza caracteristicilor date
        """
        if self.model is None:
            raise ValueError("Modelul nu a fost antrenat încă!")
        
        # Creează un DataFrame cu caracteristicile de intrare
        input_data = {
            'year': year,
            'odometer': odometer,
            'manufacturer': manufacturer,
            'model': model,
        }
        
        # Adaugă caracteristicile opționale dacă sunt furnizate
        optional_features = {
            'fuel': fuel,
            'condition': condition,
            'transmission': transmission,
            'type': vehicle_type,
            'cylinders': cylinders,
            'title_status': title_status,
            'drive': drive,
            'size': size,
            'paint_color': paint_color
        }
        
        for feature, value in optional_features.items():
            if value is not None and feature in self.categorical_columns:
                input_data[feature] = value
        
        # Completează cu valori implicite pentru caracteristicile lipsă
        for col in self.feature_columns:
            if col not in input_data:
                if col in self.categorical_columns:
                    # Folosește cea mai frecventă valoare din antrenament
                    most_common = self.label_encoders[col].classes_[0]
                    input_data[col] = most_common
                else:
                    input_data[col] = 0
        
        # Convertește în DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Aplică encoding pentru coloanele categorice
        for col in self.categorical_columns:
            if col in input_df.columns:
                try:
                    input_df[col] = self.label_encoders[col].transform(
                        input_df[col].astype(str)
                    )
                except ValueError:
                    # Dacă valoarea nu a fost văzută în antrenament, folosește prima clasă
                    input_df[col] = 0
        
        # Asigură-te că ordinea coloanelor este corectă
        input_df = input_df[self.feature_columns]
        
        # Prezice prețul
        predicted_price = self.model.predict(input_df)[0]
        
        return round(predicted_price, 2)
    
    def get_feature_importance(self):
        """
        Returnează importanța caracteristicilor
        """
        if self.model is None:
            raise ValueError("Modelul nu a fost antrenat încă!")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_path='car_price_model.pkl'):
        """
        Salvează modelul antrenat
        """
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }
        joblib.dump(model_data, model_path)
        print(f"Modelul a fost salvat în {model_path}")
    
    def load_model(self, model_path='car_price_model.pkl'):
        """
        Încarcă un model salvat
        """
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.categorical_columns = model_data['categorical_columns']
        self.numerical_columns = model_data['numerical_columns']
        print(f"Modelul a fost încărcat din {model_path}")
