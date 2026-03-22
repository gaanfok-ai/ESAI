import json
import os
from codecarbon import OfflineEmissionsTracker
from datetime import datetime
import csv

class Logger:
    def __init__(self, ds_name: str, model_name: str, train_bool=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{ds_name}_{timestamp}"
        self.ds_name = ds_name
        self.model_name = model_name
        if train_bool:
            self.description = f"Fine tuning of the {model_name} on {ds_name} dataset!"
        else:
            self.description = f"Inference of the {model_name} on {ds_name} dataset!"

        self.tracker = OfflineEmissionsTracker(
            country_iso_code="KAZ", 
            log_level="error",
            save_to_file=False,
            measure_power_secs = 1,
        )

        self.parser_acc = 0.0
        self.llm_as_judge_acc = 0.0

    def start_collecting_energy(self):
        '''Starts the background hardware polling'''
        print("Starting to collect data")
        self.tracker.start()

    def end_collecting_energy(self):
        self.tracker.stop()
        self.metrics = self.tracker.final_emissions_data
        print("Data collection has been finished!")
        print(f"Total run time: {self.metrics.duration}")

    def save_metrics(self):
        """Builds the final metrics dictionary and appends it to a JSON file."""
        # 1. Ensure the results directory exists
        os.makedirs("results", exist_ok=True)
        json_path = os.path.join("results", f"{self.ds_name}.json")
        if not hasattr(self, 'metrics'):
            print("⚠️ Warning: end_collecting_energy() must be called before save_metrics().")
            return
        run_data = {
            "experiment_id": self.experiment_id,
            "model": self.model_name,
            "dataset": self.ds_name,
            "description": self.description,
            "duration_seconds": self.metrics.duration,
            # --- ENERGY (Total volume consumed in kWh) ---
            "total_energy_kwh": self.metrics.energy_consumed,
            "cpu_energy_kwh": self.metrics.cpu_energy,
            "gpu_energy_kwh": self.metrics.gpu_energy,
            "ram_energy_kwh": self.metrics.ram_energy,
            # --- POWER (Average rate of consumption in Watts) ---
            "cpu_power_watts": self.metrics.cpu_power,
            "gpu_power_watts": self.metrics.gpu_power,
            "ram_power_watts": self.metrics.ram_power,
            "emissions_kg_co2": self.metrics.emissions,
            # --- EVALUATION METRICS ---
            "parser_acc": self.parser_acc,
            "llm_as_judge_acc": self.llm_as_judge_acc
        }
        # 4. Read existing data and append
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # If the file exists but is empty or corrupted, start fresh
                    data = [] 
        else:
            data = []
        data.append(run_data)
        # 5. Write everything back to the file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"✅ Metrics successfully saved to {json_path}")

    def save_answers(self, results):
        os.makedirs('results', exist_ok=True)
        csv_path = f"results/{self.experiment_id}.csv"
        headers = [
            "id", "question", "choices", "generated_text",
            "answerKey", "answerText",
            "parser_extracted_ans", "parser_verdict", "llm_as_judge_verdict"
        ]

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(results)

        print(f"✅ Answers saved to {csv_path}")
        return csv_path