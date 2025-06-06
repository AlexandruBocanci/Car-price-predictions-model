# Car Price Prediction Model

Acest proiect implementează un model de Machine Learning pentru predicția prețurilor mașinilor second-hand folosind Random Forest Regressor.

## Structura Proiectului

```
Car Price Model/
├── vehicles.csv                 # Dataset-ul cu datele mașinilor
├── car_price_predictor.py      # Clasa principală pentru modelul ML
├── train_model.py              # Script pentru antrenarea modelului
├── predict.py                  # Script pentru predicții interactive
├── requirements.txt            # Dependențele Python
└── README.md                   # Documentația
```

## Instalare

1. **Instalați dependențele:**
```bash
pip install -r requirements.txt
```

## Utilizare

### 1. Antrenarea Modelului

Pentru a antrena modelul pe datele din `vehicles.csv`:

```bash
python train_model.py
```

Acest script va:
- Încărca datele din CSV
- Curăța datele (elimină coloanele inutile și rândurile cu valori lipsă)
- Pregăti caracteristicile (One-Hot Encoding pentru variabilele categorice)
- Antrena un model Random Forest
- Evalua modelul folosind RMSE și MAE
- Salva modelul antrenat în `car_price_model.pkl`

### 2. Predicții Interactive

Pentru a face predicții pentru mașini individuale:

```bash
python predict.py
```

Acest script oferă:
- Interfață interactivă pentru introducerea datelor mașinii
- Predicții instant pentru prețul estimat
- Opțiune pentru predicții în lot din fișier CSV

### 3. Utilizare Programatică

```python
from car_price_predictor import CarPricePredictor

# Încarcă modelul antrenat
predictor = CarPricePredictor()
predictor.load_model('car_price_model.pkl')

# Prezice prețul unei mașini
price = predictor.predict_price(
    manufacturer='toyota',
    model='camry',
    year=2015,
    odometer=75000,
    fuel='gas',
    condition='good',
    transmission='automatic',
    vehicle_type='sedan'
)

print(f"Prețul estimat: ${price:,.2f}")
```

## Caracteristicile Modelului

### Date de Intrare Procesate:
- **Numerice:** year, odometer
- **Categorice:** manufacturer, model, fuel, condition, transmission, type, cylinders, title_status, drive, size, paint_color

### Coloane Eliminate:
- id, url, region, region_url, VIN, image_url, description, county, state, lat, long, posting_date

### Preprocesare:
- Eliminarea rândurilor cu valori lipsă
- Filtrarea valorilor extreme (preț: $500-$100,000, an: 1990-2025, kilometraj: 0-500,000)
- Label Encoding pentru variabilele categorice
- Împărțirea datelor în 80% antrenare / 20% test

### Model:
- **Algoritm:** Random Forest Regressor
- **Hiperparametri:** 100 de arbori, toate core-urile CPU
- **Evaluare:** RMSE și MAE pe setul de test

## Funcționalități Avansate

### 1. Importanța Caracteristicilor
Modelul poate afișa care caracteristici sunt cele mai importante pentru predicția prețului:

```python
importance_df = predictor.get_feature_importance()
print(importance_df)
```

### 2. Predicții Flexibile
Funcția de predicție acceptă caracteristici opționale și completează automat valorile lipsă:

```python
# Predicție cu date minime
price = predictor.predict_price(
    manufacturer='honda',
    model='civic',
    year=2018,
    odometer=60000
)

# Predicție cu toate detaliile
price = predictor.predict_price(
    manufacturer='honda',
    model='civic',
    year=2018,
    odometer=60000,
    fuel='gas',
    condition='excellent',
    transmission='manual',
    vehicle_type='sedan',
    cylinders='4',
    title_status='clean',
    drive='fwd',
    size='compact',
    paint_color='blue'
)
```

### 3. Salvare și Încărcare Model
Modelul antrenat poate fi salvat și reutilizat:

```python
# Salvare
predictor.save_model('my_model.pkl')

# Încărcare
new_predictor = CarPricePredictor()
new_predictor.load_model('my_model.pkl')
```

## Extinderea Proiectului

Proiectul este construit modular pentru a permite extinderi ușoare:

1. **Interfață Web:** Adăugați Flask/FastAPI pentru o interfață web
2. **API REST:** Creați endpoint-uri pentru predicții
3. **Modele Alternative:** Înlocuiți Random Forest cu alte algoritme
4. **Vizualizări:** Adăugați grafice pentru analiza datelor
5. **Validare Avansată:** Implementați cross-validation și grid search

## Performanța Modelului

Modelul va afișa metricile de performanță după antrenare:
- **RMSE (Root Mean Square Error):** Eroarea pătrată medie
- **MAE (Mean Absolute Error):** Eroarea absolută medie

Aceste metrici vă vor ajuta să înțelegeți cât de precis este modelul în predicțiile sale.

## Depanare

### Probleme Comune:

1. **"FileNotFoundError: car_price_model.pkl"**
   - Rulați `python train_model.py` pentru a antrena modelul

2. **"Valori categorice necunoscute"**
   - Modelul folosește valorile din setul de antrenare. Pentru valori noi, se va folosi o valoare implicită

3. **"Dataset prea mare"**
   - Pentru seturi de date foarte mari, considerați să folosiți doar o porțiune pentru testare

## Contribuții

Pentru a contribui la proiect:
1. Adăugați funcționalități noi în `car_price_predictor.py`
2. Creați scripturi suplimentare pentru cazuri de utilizare specifice
3. Îmbunătățiți documentația și exemplele
