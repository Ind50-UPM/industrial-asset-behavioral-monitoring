"""
Punto de entrada CLI para el sistema de Identificación de Estados Industriales.
Soporta entrenamiento, predicción y benchmarking de algoritmos.
"""

import sys, os
import joblib, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from typing import Callable
from src.models import StateClassifier
from src.data_processor import IndustrialDataProcessor 
from src.utils import setup_i18n

def parse_arguments(translator: Callable[[str], str]) -> argparse.Namespace:
    """Configura los argumentos de la línea de comandos con soporte i18n."""
    _ = translator
    parser = argparse.ArgumentParser(
        description=_("Framework de Identificación de Estados para Activos IIoT")
    )
    
    parser.add_argument(
        '--mode', choices=['train', 'predict'], required=True,
        help=_("Modo de ejecución: entrenar un nuevo modelo o predecir.")
    )
    parser.add_argument(
        '--algo', choices=['rf', 'xgb'], default='rf',
        help=_("Algoritmo: Random Forest (rf) o XGBoost (xgb).")
    )
    parser.add_argument(
        '--lang', default='en', choices=['es', 'en'],
        help=_("Idioma de la interfaz.")
    )
    parser.add_argument(
        '--data_ana', type=str, 
        help=_("Ruta al archivo de datos analógicos (Parquet).")
    )
    
    parser.add_argument(
        '--data_dig', type=str, 
        help=_("Ruta al archivo de datos digitales (Parquet).")
    )
    
    parser.add_argument(
        '--model_out', type=str, 
        help=_("Ruta de salida para el modelo entrenado.")
    )
    
    parser.add_argument(
        '--pred_out', type=str, 
        help=_("Ruta de salida para las predicciones.")
    )
    
    parser.add_argument(
        '--start', type=str, required=True, 
        help=_("Fecha inicio (YYYY-MM-DD HH:MM:SS).")
    )
    parser.add_argument(
        '--end', type=str, required=True, 
        help=_("Fecha fin (YYYY-MM-DD HH:MM:SS).")
    )
    parser.add_argument(
        '--output', type=str, default='models/state_model.joblib', 
        help=_("Ruta del modelo.")
    )

    return parser.parse_args()

def main() -> None:
    """Orquestador principal con soporte multilenguaje."""
    # Detección del idioma antes del parseo para traducir la ayuda del CLI
    temp_args = sys.argv
    lang = 'en'
    if '--lang' in temp_args:
        try:
            lang = temp_args[temp_args.index('--lang') + 1]
        except (IndexError, ValueError):
            pass
    
    # Configuración del traductor y argumentos
    _t = setup_i18n(lang)
    args = parse_arguments(_t)
    
    # Definición de rutas (prioriza el argumento --data del launch.json)
    analog_path = args.data_ana if args.data_ana else "../../../data/analogicas_nonans.parquet"
    digital_path = args.data_dig if args.data_dig else "../../data/digitales.parquet"

    # Inicialización del procesador
    processor = IndustrialDataProcessor(
        analog_path=analog_path,
        digital_path=digital_path
    )

    if args.mode == 'train':
        dout = args.model_out if args.model_out else "../models/Modela_A/"
        bname= os.path.basename(args.data_ana) + "_" + \
                        args.algo.upper()
        print(_t("[*] Iniciando validación cruzada: {} a {}").format(
            args.start, args.end))
        X, y = processor.get_training_data(args.start, args.end)
        
        model_wrapper = StateClassifier(model_type=args.algo, translator=_t)
        
        # Implementación de K-Fold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # Escalamos antes de validar para mantener consistencia
        X_scaled = model_wrapper.scaler.fit_transform(X)
        scores = cross_val_score(model_wrapper.clf, X_scaled, y, cv=kf)
        
        # Reporte de resultados por fold
        for i, score in enumerate(scores):
            print(_t("Fold {}: Accuracy = {:.4f}").format(i+1, score))
        
        avg_acc = scores.mean()
        std_acc = scores.std()
        print(_t("[+] Promedio: {:.4f} (+/- {:.4f})").format(avg_acc, std_acc))
        
        # Guardar resultados detallados en Excel
        rout = args.pred_out if args.pred_out else "../predictions/Modela_A/"
        
        os.makedirs(rout, exist_ok=True)
        res_df = pd.DataFrame({
            'Fold': np.arange(1, len(scores) + 1),
            'Accuracy': scores
        })
        res_file = os.path.join(rout+bname+"_" +  
                    f"metrics_{args.algo}_{args.start[:10]}.xlsx")
        res_df.to_excel(res_file, index=False)
        print(_t("[+] Resultados de métricas guardados en: {}").format(res_file))
        
        # Entrenamiento final con todos los datos y guardado
        model_wrapper.train(X, y) 
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        mes_file = dout+bname+".joblib"
        model_wrapper.save(mes_file)
        print(_t("[+] Modelo guardado en: {}").format(mes_file))

    elif args.mode == 'predict':
        print(_t("[*] Cargando modelo desde: {}").format(args.model_out))
        model_data = joblib.load(args.model_out)
        model = model_data["model"]
        scaler = model_data["scaler"]

        print(_t("[*] Cargando datos para predicción: {} a {}").format(
                            args.start, args.end))
        X_test, _ = processor.get_training_data(args.start, args.end)
        
        # Predicción con el modelo cargado
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        
        # Guardar predicciones
        bname= os.path.basename(args.data_ana) 
        pred_out = args.pred_out if args.pred_out else sys.exit(
            _t("Error: Ruta de salida para predicciones no especificada."))
        os.makedirs(pred_out, exist_ok=True)
        pred_file = os.path.join(pred_out, "predictions_{}_{}.xlsx".format(
                            bname, args.algo.upper()))
        pd.DataFrame(predictions, columns=['Predicted_State']).to_excel(
                            pred_file, index=False)
        print(_t("[+] Predicciones guardadas en: {}").format(pred_file))

if __name__ == "__main__":
    main()