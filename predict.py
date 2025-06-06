from car_price_predictor import CarPricePredictor

def predict_single_car():
    """
    Funcție simplă pentru predicția prețului unei singure mașini
    """
    # Încarcă modelul antrenat
    predictor = CarPricePredictor()
    
    try:
        predictor.load_model('car_price_model.pkl')
        print("Modelul a fost încărcat cu succes!")
    except FileNotFoundError:
        print("Eroare: Modelul nu a fost găsit. Rulați train_model.py mai întâi.")
        return
    
    print("\n" + "="*50)
    print("PREDICTOR PREȚ MAȘINĂ SECOND-HAND")
    print("="*50)
    
    # Colectează datele de la utilizator
    try:
        manufacturer = input("Introduceti producatorul (ex: toyota, ford, honda): ").strip().lower()
        model = input("Introduceti modelul (ex: camry, f-150, civic): ").strip().lower()
        year = int(input("Introduceti anul (ex: 2015): "))
        odometer = int(input("Introduceti kilometrajul (ex: 75000): "))
        
        print("\nCaracteristici opționale (apăsați Enter pentru a sări):")
        fuel = input("Combustibil (gas/diesel/electric): ").strip().lower() or None
        condition = input("Starea (excellent/good/fair/poor): ").strip().lower() or None
        transmission = input("Transmisia (automatic/manual): ").strip().lower() or None
        vehicle_type = input("Tipul vehiculului (sedan/suv/pickup/coupe/hatchback): ").strip().lower() or None
        cylinders = input("Numărul de cilindri (4/6/8): ").strip() or None
        title_status = input("Statusul titlului (clean/lien/salvage): ").strip().lower() or None
        drive = input("Tracțiunea (fwd/rwd/4wd): ").strip().lower() or None
        size = input("Mărimea (compact/mid-size/full-size): ").strip().lower() or None
        paint_color = input("Culoarea (black/white/silver/red/blue): ").strip().lower() or None
        
        # Fă predicția
        predicted_price = predictor.predict_price(
            manufacturer=manufacturer,
            model=model,
            year=year,
            odometer=odometer,
            fuel=fuel,
            condition=condition,
            transmission=transmission,
            vehicle_type=vehicle_type,
            cylinders=cylinders,
            title_status=title_status,
            drive=drive,
            size=size,
            paint_color=paint_color
        )
        
        print("\n" + "="*50)
        print("REZULTATUL PREDICȚIEI")
        print("="*50)
        print(f"Mașina: {manufacturer.title()} {model.title()} {year}")
        print(f"Kilometraj: {odometer:,} mile")
        if condition:
            print(f"Starea: {condition.title()}")
        if fuel:
            print(f"Combustibil: {fuel.title()}")
        if transmission:
            print(f"Transmisia: {transmission.title()}")
        
        print(f"\n💰 PREȚUL ESTIMAT: ${predicted_price:,.2f}")
        print("="*50)
        
    except ValueError as e:
        print(f"Eroare: Vă rugăm să introduceți valori valide pentru anul și kilometrajul.")
    except Exception as e:
        print(f"Eroare neașteptată: {str(e)}")

def batch_predict():
    """
    Funcție pentru predicții în lot din fișier CSV
    """
    predictor = CarPricePredictor()
    
    try:
        predictor.load_model('car_price_model.pkl')
        print("Modelul a fost încărcat cu succes!")
    except FileNotFoundError:
        print("Eroare: Modelul nu a fost găsit. Rulați train_model.py mai întâi.")
        return
    
    import pandas as pd
    
    # Exemplu de fișier CSV de input
    input_file = input("Introduceți calea către fișierul CSV cu mașinile de evaluat: ")
    
    try:
        df = pd.read_csv(input_file)
        print(f"Încărcat {len(df)} mașini pentru evaluare...")
        
        predictions = []
        for _, row in df.iterrows():
            try:
                price = predictor.predict_price(
                    manufacturer=row.get('manufacturer', ''),
                    model=row.get('model', ''),
                    year=row.get('year', 2015),
                    odometer=row.get('odometer', 100000),
                    fuel=row.get('fuel', None),
                    condition=row.get('condition', None),
                    transmission=row.get('transmission', None),
                    vehicle_type=row.get('type', None)
                )
                predictions.append(price)
            except:
                predictions.append(None)
        
        df['predicted_price'] = predictions
        output_file = 'predictions_output.csv'
        df.to_csv(output_file, index=False)
        print(f"Predicțiile au fost salvate în {output_file}")
        
    except FileNotFoundError:
        print("Fișierul specificat nu a fost găsit.")
    except Exception as e:
        print(f"Eroare: {str(e)}")

if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        print("MENIU PRINCIPAL")
        print("="*50)
        print("1. Predicție pentru o singură mașină")
        print("2. Predicții în lot din fișier CSV")
        print("3. Ieșire")
        
        choice = input("\nAlegeți o opțiune (1-3): ").strip()
        
        if choice == '1':
            predict_single_car()
        elif choice == '2':
            batch_predict()
        elif choice == '3':
            print("La revedere!")
            break
        else:
            print("Opțiune invalidă. Vă rugăm să alegeți 1, 2 sau 3.")
