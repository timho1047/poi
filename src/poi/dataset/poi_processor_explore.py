"""
POI Data Processing Module
Process raw POI data and generate various files required for training and testing
"""

import os
import random
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Dict, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download, upload_file
from openlocationcode import openlocationcode as olc


class POIProcessor:
    """POI Data Processor"""

    def __init__(self, dataset_name: str, hf_org: str = "comp5331poi"):
        self.dataset_name = dataset_name
        self.hf_org = hf_org
        self.hf_repo = f"{hf_org}/{dataset_name}"

    def download_from_hf(self, filename: str, local_path: str) -> str:
        """Download file from Hugging Face"""
        try:
            downloaded_path = hf_hub_download(repo_id=self.hf_repo, filename=filename, repo_type="dataset")
            return downloaded_path
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            raise

    def upload_to_hf(self, local_path: str, hf_path: str) -> None:
        """Upload file to Hugging Face"""
        try:
            upload_file(path_or_fileobj=local_path, path_in_repo=hf_path, repo_id=self.hf_repo, repo_type="dataset")
            print(f"✅ Uploaded {local_path} to {hf_path}")
        except Exception as e:
            print(f"❌ Error uploading {hf_path}: {e}")
            raise

    def filter_data(self, df: pd.DataFrame, poi_min_freq: int = 10, user_min_freq: int = 10) -> pd.DataFrame:
        """Filter low-frequency users and POIs"""
        df = df.copy()

        # Calculate POI frequency
        df["PoiFreq"] = df.groupby("Pid")["Uid"].transform("count")
        df = df[df["PoiFreq"] >= poi_min_freq]

        # Calculate user frequency
        df["UserFreq"] = df.groupby("Uid")["Pid"].transform("count")
        df = df[df["UserFreq"] >= user_min_freq]

        df = df.drop(columns=["PoiFreq", "UserFreq"])
        return df

    def get_pluscode(self, latitude: float, longitude: float) -> str:
        """Generate Plus Code"""
        plus_code = olc.encode(latitude, longitude)
        return plus_code[:6]

    def safe_pluscode(self, latitude: float, longitude: float) -> str:
        """Safe Plus Code generation"""
        if pd.isna(latitude) or pd.isna(longitude):
            return pd.NA
        return olc.encode(latitude, longitude)[:6]

    def process_raw_data(self, poi_min_freq: int = 10, user_min_freq: int = 10) -> Tuple[pd.DataFrame, str]:
        """Process raw CSV data"""
        print(f"🔄 Processing raw data for {self.dataset_name}...")

        # Download raw data from HF
        raw_data_path = self.download_from_hf(f"{self.dataset_name} base.csv", f"/tmp/{self.dataset_name}_base.csv")

        # Read data
        df = pd.read_csv(raw_data_path, sep=",", encoding="latin-1", header=0)

        # Process coordinates
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        valid_rows = df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)
        df = df[valid_rows].copy()

        # Generate Plus Code
        df["Region"] = df.apply(lambda row: self.safe_pluscode(row["Latitude"], row["Longitude"]), axis=1)
        df = df.dropna(subset=["Region"])

        # Process time
        df["UTC Time"] = pd.to_datetime(df["UTC Time"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
        
        ## Update
        df["year_month"] = df["UTC Time"].dt.to_period("M")
        latest_month = df.sort_values("UTC Time", ascending=False)["year_month"].iloc[0]
        print("Latest month is:", latest_month)
        df["Visit Recency"] = (latest_month - df["year_month"]).apply(lambda x: x.n if x.n<6 else 100)
        # 100 means no visit in latest 6 months, 0~5 means visits in latest x+1 months
        
        df["Timezone Offset"] = pd.to_numeric(df["Timezone Offset"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["UTC Time", "Timezone Offset"])

        df["Local Time"] = (df["UTC Time"] + df["Timezone Offset"].apply(lambda x: timedelta(minutes=int(x)))).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Rename columns
        df.columns = ["Uid", "Pid", "Venue Category ID", "Catname", "Lat", "Lon", "year_month", "Visit Recency", "Timezone Offset", "UTC Time", "Region", "Time"]
        df = df[["Uid", "Pid", "Catname", "Region", "Time", "Visit Recency"]]

        # Filter data
        filtered_df = self.filter_data(df, poi_min_freq, user_min_freq)

        print(f"✅ Processed {len(filtered_df)} records")
        return filtered_df, raw_data_path

    def create_mappings(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Create ID mappings"""
        print("🔄 Creating ID mappings...")

        uids = list(df["Uid"].unique())
        pids = list(df["Pid"].unique())
        cats = list(df["Catname"].unique())
        regs = list(df["Region"].unique())

        # Use 2024 as random seed to maintain team consistency
        random.seed(2024)
        random.shuffle(uids)
        random.shuffle(pids)
        random.shuffle(cats)
        random.shuffle(regs)

        uid_map = {uid: i for i, uid in enumerate(uids, start=1)}
        pid_map = {pid: i for i, pid in enumerate(pids, start=1)}
        cat_map = {cat: i for i, cat in enumerate(cats, start=1)}
        reg_map = {reg: i for i, reg in enumerate(regs, start=1)}

        # Apply mappings
        df["Uid"] = df["Uid"].map(uid_map)
        df["Pid"] = df["Pid"].map(pid_map)
        df["Catname"] = df["Catname"].map(cat_map)
        df["Region"] = df["Region"].map(reg_map)

        mappings = {"uid_map": uid_map, "pid_map": pid_map, "cat_map": cat_map, "reg_map": reg_map}

        print("✅ Created ID mappings")
        return df, mappings

    def create_poi_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create POI information"""
        print("🔄 Creating POI info...")

        # Process time - keep full datetime instead of just hour
        df["Time"] = pd.to_datetime(df["Time"])

        # Create POI sequence
        poi_sequence = df.groupby("Uid").agg({"Pid": list, "Catname": list}).reset_index()

        # Get neighbor information
        poi_neighbors = self._get_neighbors(poi_sequence, "Pid", 1)
        forward_neighbors = self._get_forward_neighbors(poi_sequence, "Pid", 1)

        # Create POI information
        ## Update
        results = []
        for pid, group in df.groupby("Pid"):
            # Keep only the latest visit of each user
            group_deduped = group.sort_values("Visit Recency", ascending=True).drop_duplicates(subset=["Uid"], keep='first')
            uid_list = group_deduped["Uid"].tolist()
            recency_list = group_deduped["Visit Recency"].tolist()
            
            catname = group["Catname"].iloc[0]
            region = group["Region"].iloc[0]
            time_list = group["Time"].tolist()
            
            results.append({
                "Pid": pid,
                "Uid": uid_list,
                "Visit Recency": recency_list,
                "Catname": catname,
                "Region": region,
                "Time": time_list
            })
        poi_info = pd.DataFrame(results)
        
        # Process user and time lists
        # Uid上一步已去重
        poi_info["Time"] = poi_info["Time"].apply(lambda times: [time for time, count in Counter(times).items() if count >= 1])

        # Add neighbor information
        poi_info["neighbors"] = poi_info["Pid"].map(poi_neighbors.set_index("Pid")["neighbors"])
        poi_info["forward_neighbors"] = poi_info["Pid"].map(forward_neighbors.set_index("Pid")["neighbors"])

        print("✅ Created POI info")
        return poi_info

    def _get_forward_neighbors(self, df: pd.DataFrame, column: str, min_freq: int = 1) -> pd.DataFrame:
        """Get forward neighbors"""
        neighbor_counts = defaultdict(Counter)
        all_pois = set()

        for sequence in df[column]:
            all_pois.update(sequence)
            for i in range(len(sequence) - 1):
                current_poi = sequence[i]
                next_poi = sequence[i + 1]
                neighbor_counts[current_poi][next_poi] += 1

        df_data = []
        for poi in all_pois:
            counter = neighbor_counts.get(poi, {})
            filtered_neighbors = {neighbor: freq for neighbor, freq in counter.items() if freq >= min_freq}
            if filtered_neighbors:
                sorted_neighbors = [neighbor for neighbor, _ in sorted(filtered_neighbors.items(), key=lambda x: x[1], reverse=True)]
            else:
                sorted_neighbors = []
            df_data.append((poi, sorted_neighbors))

        neighbors_df = pd.DataFrame(df_data, columns=[column, "neighbors"])
        return neighbors_df

    def _get_neighbors(self, df: pd.DataFrame, column: str, min_freq: int = 1) -> pd.DataFrame:
        """Get neighbors"""
        neighbor_counts = defaultdict(Counter)
        all_pois = set()

        for sequence in df[column]:
            all_pois.update(sequence)
            for i, poi in enumerate(sequence):
                if i > 0:
                    neighbor_counts[poi][sequence[i - 1]] += 1
                if i < len(sequence) - 1:
                    neighbor_counts[poi][sequence[i + 1]] += 1

        df_data = []
        for poi in all_pois:
            counter = neighbor_counts.get(poi, {})
            filtered_neighbors = {neighbor: freq for neighbor, freq in counter.items() if freq >= min_freq}
            sorted_neighbors = [neighbor for neighbor, _ in sorted(filtered_neighbors.items(), key=lambda x: x[1], reverse=True)]
            df_data.append((poi, sorted_neighbors))

        neighbors_df = pd.DataFrame(df_data, columns=[column, "neighbors"])
        return neighbors_df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split training and test data"""
        print("🔄 Splitting data...")

        df = df[["Uid", "Pid", "Time"]]
        df = df.sort_values(by="Time")

        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        test_df = df[train_size:]

        # Ensure users and POIs in test set also exist in training set
        def remove_users_pois_test(df_train, df_test):
            users_train = df_train["Uid"].unique()
            pois_train = df_train["Pid"].unique()
            df_test = df_test[df_test["Uid"].isin(users_train)]
            df_test = df_test[df_test["Pid"].isin(pois_train)]
            return df_test

        test_df = remove_users_pois_test(train_df, test_df)

        print(f"✅ Split data: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df

    def generate_sequences(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, window_size: int = 50, step_size: int = 10, mask_prob: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """Generate sequence data"""
        print("🔄 Generating sequences...")

        train = self._generate_train_sequences(train_df, window_size, step_size, mask_prob)
        val, test = self._generate_test_sequences(test_df, window_size)

        # Generate test sequences for all data
        all_data = pd.concat([train_df, test_df])
        val_all, test_all = self._generate_test_sequences(all_data, window_size)

        sequences = {"train": train, "val": val, "test": test, "test_all": test_all}

        print("✅ Generated sequences")
        return sequences

    def _generate_train_sequences(self, df: pd.DataFrame, window_size: int, step_size: int, mask_prob: float) -> pd.DataFrame:
        """Generate training sequences"""
        df = df.copy()
        df["Time"] = pd.to_datetime(df["Time"])

        results = []

        for uid, group in df.groupby("Uid"):
            group = group.sort_values("Time").reset_index(drop=True)

            if len(group) > 80:
                group = group.iloc[-80:]

            n = len(group)

            if n < window_size:
                if n >= 10:
                    input_pids = group["Pid"].iloc[:-1].tolist()
                    input_times = group["Time"].iloc[:-1].tolist()
                    target_pid = group["Pid"].iloc[-1]
                    target_time = group["Time"].iloc[-1]

                    results.append({"Uid": uid, "Pids": input_pids, "Times": input_times, "Target": target_pid, "Target_time": target_time})
                continue

            for start in range(n - 1, window_size - 2, -step_size):
                end = start + 1
                window = group.iloc[start - window_size + 1 : start + 1]

                input_pids = window["Pid"].iloc[:-1].tolist()
                input_times = window["Time"].iloc[:-1].tolist()
                original_target_pid = window["Pid"].iloc[-1]
                original_target_time = window["Time"].iloc[-1]

                if random.random() < mask_prob and len(input_pids) >= 1:
                    drop_idx = random.randint(0, len(input_pids) - 1)
                    target_pid = input_pids[drop_idx]
                    target_time = input_times[drop_idx]
                    input_pids = input_pids[:drop_idx] + input_pids[drop_idx + 1 :] + [original_target_pid]
                    input_times = input_times[:drop_idx] + input_times[drop_idx + 1 :] + [original_target_time]
                else:
                    target_pid = original_target_pid
                    target_time = original_target_time

                results.append({"Uid": uid, "Pids": input_pids, "Times": input_times, "Target": target_pid, "Target_time": target_time})

        train = pd.DataFrame(results)
        train["Times"] = train["Times"].apply(lambda x: [t.strftime("%Y-%m-%d %H:%M:%S") for t in x])
        train["Target_time"] = train["Target_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        return train

    def _generate_test_sequences(self, test_df: pd.DataFrame, window_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate test sequences"""
        test_df = test_df.copy()
        test_df["Time"] = pd.to_datetime(test_df["Time"])

        val_records = []
        test_records = []

        for uid, group in test_df.groupby("Uid"):
            group = group.sort_values("Time").reset_index(drop=True)
            n = len(group)

            if n < window_size:
                if n > 2:
                    test_records.append(
                        {
                            "Uid": uid,
                            "Pids": group["Pid"].iloc[:-1].tolist(),
                            "Times": group["Time"].iloc[:-1].tolist(),
                            "Target": group["Pid"].iloc[-1],
                            "Target_time": group["Time"].iloc[-1],
                        }
                    )
                    val_records.append(
                        {
                            "Uid": uid,
                            "Pids": group["Pid"].iloc[:-2].tolist(),
                            "Times": group["Time"].iloc[:-2].tolist(),
                            "Target": group["Pid"].iloc[-2],
                            "Target_time": group["Time"].iloc[-2],
                        }
                    )
                continue

            if n >= window_size + 1:
                val_start = n - window_size - 1
                val_window = group.iloc[val_start : val_start + window_size]
                val_records.append(
                    {
                        "Uid": uid,
                        "Pids": val_window["Pid"].iloc[:-1].tolist(),
                        "Times": val_window["Time"].iloc[:-1].tolist(),
                        "Target": val_window["Pid"].iloc[-1],
                        "Target_time": val_window["Time"].iloc[-1],
                    }
                )

            test_window = group.iloc[n - window_size :]
            test_records.append(
                {
                    "Uid": uid,
                    "Pids": test_window["Pid"].iloc[:-1].tolist(),
                    "Times": test_window["Time"].iloc[:-1].tolist(),
                    "Target": test_window["Pid"].iloc[-1],
                    "Target_time": test_window["Time"].iloc[-1],
                }
            )

        val_df = pd.DataFrame(val_records)
        test_df = pd.DataFrame(test_records)

        for df in [val_df, test_df]:
            df["Times"] = df["Times"].apply(lambda x: [t.strftime("%Y-%m-%d %H:%M:%S") for t in x])
            df["Target_time"] = df["Target_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        return val_df, test_df

    def process_dataset(
        self, poi_min_freq: int = 10, user_min_freq: int = 10, window_size: int = 50, step_size: int = 10, mask_prob: float = 0.1
    ) -> None:
        """Complete data processing workflow"""
        print(f"🚀 Starting dataset processing for {self.dataset_name}")

        # 1. Process raw data
        filtered_df, raw_data_path = self.process_raw_data(poi_min_freq, user_min_freq)

        # 2. Create mappings
        df, mappings = self.create_mappings(filtered_df)

        # 3. Create POI information
        poi_info = self.create_poi_info(df)

        # 4. Split data
        train_df, test_df = self.split_data(df)

        # 5. Generate sequences
        sequences = self.generate_sequences(train_df, test_df, window_size, step_size, mask_prob)

        # 6. Save to temporary directory
        temp_dir = f"/tmp/{self.dataset_name}_processed"
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(f"{temp_dir}/data", exist_ok=True)

        # Save mapping files
        pd.DataFrame(list(mappings["uid_map"].items()), columns=["Original_Uid", "Mapped_Uid"]).to_csv(f"{temp_dir}/uid_mapping.csv", index=False)
        pd.DataFrame(list(mappings["pid_map"].items()), columns=["Original_Pid", "Mapped_Pid"]).to_csv(f"{temp_dir}/pid_mapping.csv", index=False)
        pd.DataFrame(list(mappings["cat_map"].items()), columns=["Original_Catname", "Mapped_Catname"]).to_csv(
            f"{temp_dir}/catname_mapping.csv", index=False
        )
        pd.DataFrame(list(mappings["reg_map"].items()), columns=["Original_Region", "Mapped_Region"]).to_csv(
            f"{temp_dir}/region_mapping.csv", index=False
        )

        # Save data files
        df.to_csv(f"{temp_dir}/data.csv", index=False)
        poi_info.to_csv(f"{temp_dir}/poi_info.csv", index=False)
        train_df.to_csv(f"{temp_dir}/train_data.csv", index=False)
        test_df.to_csv(f"{temp_dir}/test_data.csv", index=False)

        # Save sequence files
        sequences["train"].to_csv(f"{temp_dir}/data/train.csv", index=False)
        sequences["val"].to_csv(f"{temp_dir}/data/val.csv", index=False)
        sequences["test"].to_csv(f"{temp_dir}/data/test.csv", index=False)
        sequences["test_all"].to_csv(f"{temp_dir}/data/test_all.csv", index=False)

        # 7. Upload to HF
        print("🔄 Uploading files to Hugging Face...")

        # Upload to Intermediate Files folder
        intermediate_files = [
            "uid_mapping.csv",
            "pid_mapping.csv",
            "catname_mapping.csv",
            "region_mapping.csv",
            "data.csv",
            "poi_info.csv",
            "train_data.csv",
            "test_data.csv",
        ]

        for filename in intermediate_files:
            self.upload_to_hf(f"{temp_dir}/{filename}", f"Intermediate Files/{filename}")

        # Upload files in data folder
        data_files = ["train.csv", "val.csv", "test.csv", "test_all.csv"]
        for filename in data_files:
            self.upload_to_hf(f"{temp_dir}/data/{filename}", f"Intermediate Files/data/{filename}")

        # Upload raw data to Intermediate Files
        self.upload_to_hf(raw_data_path, f"Intermediate Files/{self.dataset_name}.csv")

        print(f"✅ Dataset processing completed for {self.dataset_name}")
        print(f"📁 Files uploaded to: {self.hf_repo}/Intermediate Files/")

        # Clean up temporary files
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        # Clean up raw data file (if exists)
        if os.path.exists(raw_data_path):
            os.remove(raw_data_path)
