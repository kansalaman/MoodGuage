from config import model_config
import pandas as pd
def load_data(batched=True, test=False):
    bs=model_config['batch_size']
    df: pd.DataFrame
    