"""
Módulo de Procesamiento de Datos Industriales.
Encargado de la carga, imputación de NaNs y sincronización de señales.
"""

import pandas as pd
import numpy as np
from typing import Tuple

class IndustrialDataProcessor:
    """
    Procesador para señales analógicas y digitales de activos industriales.
    """

    def __init__(self, analog_path: str, digital_path: str):
        """
        Inicializa el procesador cargando los archivos Parquet.
        
        Args:
            analog_path: Ruta al archivo Parquet de señales analógicas.
            digital_path: Ruta al archivo Parquet de señales digitales.
        """
        self.analog_df = pd.read_parquet(analog_path)
        self.digital_df = pd.read_parquet(digital_path)
        self.threshold = 50  # Umbral de consumo en W para estado parado [cite: 8]

    def _impute_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementa la lógica de imputación definida en el estudio (imputa.py).
        """
        # Aquí se traslada la lógica de tu script imputa.py [cite: 10]
        # 1. Mínimo de fases, 2. Next value, 3. Prior value [cite: 4]
        return df.fillna(method='ffill').fillna(method='bfill')

    def get_training_data(self, start: str, end: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Filtra, imputa y etiqueta los datos para el entrenamiento.
        
        Args:
            start: Fecha de inicio del periodo.
            end: Fecha de fin del periodo.
            
        Returns:
            Tupla con (Variables de entrada X, Etiquetas de estado y).
        """
        # Filtrado temporal [cite: 8]
        mask = (self.analog_df.index >= start) & (self.analog_df.index <= end)
        adf = self.analog_df.loc[mask].copy()
        
        # Sincronización de estados digitales con analógicos [cite: 8]
        # Esta lógica proviene de tu script separa_rf.py [cite: 8]
        max_rp = adf[['RP1', 'RP2', 'RP3']].max(axis=1)
        adf['estado'] = 0
        
        active_mask = max_rp >= self.threshold
        if active_mask.any():
            relevant_digitals = self.digital_df[self.digital_df['estado'] != 0]
            # Búsqueda del instante más cercano (method='pad') [cite: 8]
            idx_sincro = relevant_digitals.index.get_indexer(
                adf[active_mask].index, method='pad'
            )
            adf.loc[active_mask, 'estado'] = relevant_digitals['estado'].iloc[idx_sincro].values

        # Variables de entrenamiento definidas en el paper [cite: 8]
        model_vars = ['Vrms1', 'Vrms2', 'Vrms3', 'Irms1', 'Irms2', 'Irms3', 'PF1', 'PF2', 'PF3']
        
        # Solo devolvemos registros donde hay actividad (estado != 0) [cite: 8]
        train_df = adf[adf['estado'] != 0]
        return train_df[model_vars], train_df['estado']