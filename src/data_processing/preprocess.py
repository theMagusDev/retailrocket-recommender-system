import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os
import logging

from src.data_processing.data_load import get_categories, get_events, get_item_properties

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
preprocessing_logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(
        self,
        data_dir="./data/raw/",
        processed_dir="./data/processed/",
        activity_threshold=1,
        weights: dict = {"view": 1, "addtocart": 2, "transaction": 3},
    ):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        self.activity_threshold = activity_threshold
        self.weights = weights
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.events = None
        self.item_props = None
        self.categories = None
        self.events_filtered = None
        self.latest_props = None

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw data."""
        preprocessing_logger.info("Loading events...")
        self.events = get_events()
        
        preprocessing_logger.info("Loading item properties...")
        self.item_props = get_item_properties()
        
        preprocessing_logger.info("Loading categories...")
        self.categories = get_categories()
        
        return self.events, self.item_props, self.categories

    def preprocess_events(self):
        """Preprocess events: timestamp to dt, add features."""
        if self.events is None:
            raise ValueError("Load data first.")
        self.events["datetime"] = pd.to_datetime(self.events["timestamp"], unit="ms")
        self.events["date"] = self.events["datetime"].dt.date
        self.events["day_of_week"] = self.events["datetime"].dt.dayofweek
        self.events["hour"] = self.events["datetime"].dt.hour
        preprocessing_logger.info("Events preprocessed.")
        return self.events

    def preprocess_props(self) -> pd.DataFrame:
        """Preprocess item properties: timestamp to dt, merge snapshots (latest per property/item)."""
        if self.item_props is None:
            raise ValueError("Load data first.")
        self.item_props["datetime"] = pd.to_datetime(self.item_props["timestamp"], unit="ms")
        self.item_props = self.item_props.sort_values(["itemid", "timestamp"])
        
        # Merge snapshots: Take the latest value per property/item
        self.latest_props = self.item_props.sort_values("timestamp", ascending=False).drop_duplicates(["itemid", "property"])
        preprocessing_logger.info("Item properties preprocessed and snapshots merged.")
        return self.latest_props

    def filter_active(self):
        """Filter active users/items based on threshold."""
        if self.events is None:
            raise ValueError("Preprocess events first.")
        
        user_activity = self.events.groupby("visitorid").size()
        active_users = user_activity[user_activity > self.activity_threshold].index
        
        item_activity = self.events.groupby("itemid").size()
        active_items = item_activity[item_activity > self.activity_threshold].index
        
        self.events_filtered = self.events[self.events["visitorid"].isin(active_users) & self.events["itemid"].isin(active_items)]
        preprocessing_logger.info("Filtered to %d interactions (%d users, %d items).", len(self.events_filtered), len(active_users), len(active_items))
        return self.events_filtered

    def fit(self):
        """Fit encoders on filtered data."""
        if self.events_filtered is None:
            raise ValueError("Filter active first.")
        
        self.user_encoder.fit(self.events_filtered["visitorid"])
        self.item_encoder.fit(self.events_filtered["itemid"])
        preprocessing_logger.info("Encoders fitted.")
        return self

    def transform(self, df=None, build_matrix=False):
        """Transform data: Encode ids, optional build sparse matrix with weights."""
        if df is None:
            df = self.events_filtered
        if df is None:
            raise ValueError("No data to transform.")
        
        df_encoded = df.copy()
        df_encoded["visitorid_encoded"] = self.user_encoder.transform(df_encoded["visitorid"])
        df_encoded["itemid_encoded"] = self.item_encoder.transform(df_encoded["itemid"])
        
        if build_matrix:
            # Map events to weights
            df_encoded["weight"] = df_encoded["event"].map(self.weights).fillna(0)
            
            # Sparse matrix: Max weight per user-item (if multiple events, take highest e.g., transaction>view)
            agg_weights = df_encoded.groupby(["visitorid_encoded", "itemid_encoded"])["weight"].max().reset_index()
            
            n_users = len(self.user_encoder.classes_)
            n_items = len(self.item_encoder.classes_)
            matrix = coo_matrix(
                (agg_weights["weight"], (agg_weights["visitorid_encoded"], agg_weights["itemid_encoded"])),
                shape=(n_users, n_items),
            )
            preprocessing_logger.info("Sparse matrix built: %s, density %.4f %%", matrix.shape, matrix.nnz / (n_users * n_items) * 100)
            return matrix
        
        return df_encoded

    def split_data(self, train_ratio=0.8):
        """Time-based split: Sort by date, first 80% train."""
        if self.events_filtered is None:
            raise ValueError("Filter active first.")
        
        sorted_events = self.events_filtered.sort_values("date")
        split_idx = int(len(sorted_events) * train_ratio)
        train = sorted_events.iloc[:split_idx]
        test = sorted_events.iloc[split_idx:]
        preprocessing_logger.info("Split: Train %d (%d users), Test %d (%d users)", len(train), len(train["visitorid"].unique()), len(test), len(test["visitorid"].unique()))
        return train, test

    def save_processed(self):
        """Save processed data to pkl."""
        if self.events_filtered is not None:
            self.events_filtered.to_pickle(self.processed_dir + "events_filtered.pkl")
            preprocessing_logger.info("Saved filtered events to %s", self.processed_dir + "events_filtered.pkl")
        if self.latest_props is not None:
            self.latest_props.to_pickle(self.processed_dir + "latest_item_props.pkl")
            preprocessing_logger.info("Saved latest props to %s", self.processed_dir + "latest_item_props.pkl")
        if self.categories is not None:
            self.categories.to_pickle(self.processed_dir + "categories.pkl")
            preprocessing_logger.info("Saved categories to %s", self.processed_dir + "categories.pkl")
        preprocessing_logger.info("Processed data saved.")
