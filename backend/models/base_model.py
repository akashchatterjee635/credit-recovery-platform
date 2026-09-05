from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseRiskAdapter(ABC):
    """
    Common contract for risk models.
    """
    
    @abstractmethod
    def predict_risk(self, applicant: pd.DataFrame) -> np.ndarray:
        """
        Predict the probability of default for the given applicant(s).
        
        Args:
            applicant (pd.DataFrame): DataFrame containing applicant state(s).
            
        Returns:
            np.ndarray: 1D array of calibrated probabilities.
        """
        pass
