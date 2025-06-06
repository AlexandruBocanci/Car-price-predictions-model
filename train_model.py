from car_price_predictor import CarPricePredictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """
    Funcția principală pentru antrenarea și evaluarea modelului
    """
    # Inițializează predictorul
    predictor = CarPricePredictor()
    
    try:
        # Încarcă datele
        predictor.load_data('vehicles.csv')
        
        # Curăță datele
        predictor.clean_data()
        
        # Pregătește caracteristicile
        X, y = predictor.prepare_features()
        
        # Antrenează modelul
        results = predictor.train_model(X, y)
        
        # Afișează importanța caracteristicilor
        print("\n" + "="*50)
        print("IMPORTANȚA CARACTERISTICILOR:")
        print("="*50)
        importance_df = predictor.get_feature_importance()
        print(importance_df.head(10))
        
        # Salvează modelul
        predictor.save_model()
        
        # Testează predicția cu exemple
        print("\n" + "="*50)
        print("EXEMPLE DE PREDICȚII:")
        print("="*50)
        
        # Exemplu 1: Toyota Camry 2015
        price1 = predictor.predict_price(
            manufacturer='toyota',
            model='camry',
            year=2015,
            odometer=75000,
            fuel='gas',
            condition='good',
            transmission='automatic',
            vehicle_type='sedan'
        )
        print(f"Toyota Camry 2015, 75k miles, good condition: ${price1:,.2f}")
        
        # Exemplu 2: Ford F-150 2018
        price2 = predictor.predict_price(
            manufacturer='ford',
            model='f-150',
            year=2018,
            odometer=50000,
            fuel='gas',
            condition='excellent',
            transmission='automatic',
            vehicle_type='pickup'
        )
        print(f"Ford F-150 2018, 50k miles, excellent condition: ${price2:,.2f}")
        
        # Exemplu 3: Honda Civic 2020
        price3 = predictor.predict_price(
            manufacturer='honda',
            model='civic',
            year=2020,
            odometer=25000,
            fuel='gas',
            condition='like new',
            transmission='manual',
            vehicle_type='sedan'
        )
        print(f"Honda Civic 2020, 25k miles, like new condition: ${price3:,.2f}")
        
        print("\n" + "="*50)
        print("ANTRENAMENT FINALIZAT CU SUCCES!")
        print("="*50)
        print("Modelul a fost salvat ca 'car_price_model.pkl'")
        print("Poți folosi funcția predict_price() pentru predicții noi")
        
    except Exception as e:
        print(f"Eroare în timpul antrenării: {str(e)}")
        return False
    
    return True

def demo_predictions():
    """
    Demonstrație a funcției de predicție
    """
    # Încarcă modelul salvat
    predictor = CarPricePredictor()
    try:
        predictor.load_model()
        
        print("\n" + "="*50)
        print("DEMONSTRAȚIE PREDICȚII INTERACTIVE")
        print("="*50)
        
        while True:
            print("\nIntroduceți datele mașinii (sau 'quit' pentru ieșire):")
            manufacturer = input("Producător (ex: toyota, ford, honda): ").strip().lower()
            if manufacturer == 'quit':
                break
                
            model = input("Model (ex: camry, f-150, civic): ").strip().lower()
            year = int(input("Anul (ex: 2015): "))
            odometer = int(input("Kilometraj (ex: 75000): "))
            fuel = input("Combustibil (gas, diesel, electric) [opțional]: ").strip().lower() or None
            condition = input("Starea (excellent, good, fair) [opțional]: ").strip().lower() or None
            transmission = input("Transmisia (automatic, manual) [opțional]: ").strip().lower() or None
            vehicle_type = input("Tipul (sedan, suv, pickup) [opțional]: ").strip().lower() or None
            
            try:
                price = predictor.predict_price(
                    manufacturer=manufacturer,
                    model=model,
                    year=year,
                    odometer=odometer,
                    fuel=fuel,
                    condition=condition,
                    transmission=transmission,
                    vehicle_type=vehicle_type
                )
                print(f"\n💰 Prețul estimat: ${price:,.2f}")
            except Exception as e:
                print(f"Eroare la predicție: {str(e)}")
                
    except FileNotFoundError:
        print("Modelul nu a fost găsit. Rulați mai întâi main() pentru antrenare.")

if __name__ == "__main__":
    # Antrenează modelul
    success = main()
    
    if success:
        # Demonstrație interactivă
        demo_predictions()
