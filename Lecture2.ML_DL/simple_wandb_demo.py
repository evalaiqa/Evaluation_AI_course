# simple_wandb_demo.py
# =============================================================================
# PYTORCH LIGHTNING + W&B — РЕАЛЬНЫЙ ДАТАСЕТ О ЖИЛЬЕ (реальные цены на графиках)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import wandb


class HousingDataModule(pl.LightningDataModule):
    """DataModule PyTorch Lightning для данных о жилье"""

    def __init__(self, batch_size=256, test_size=0.2, val_size=0.25):
        super().__init__()
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def prepare_data(self):
        """Загружает данные один раз"""
        print("📊 Loading the real California Housing dataset...")
        housing = fetch_california_housing()
        self.X, self.y = housing.data, housing.target
        print(f"📋 Dataset size: {self.X.shape}")
        print("📋 Target variable: house values in hundreds of thousands of dollars (x100k)")

    def setup(self, stage=None):
        """Готовит данные для обучения/валидации/тестирования"""
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=self.val_size, random_state=42
        )

        # Нормализация
        self.X_train = self.scaler_X.fit_transform(self.X_train)
        self.X_val = self.scaler_X.transform(self.X_val)
        self.X_test = self.scaler_X.transform(self.X_test)

        self.y_train = self.scaler_y.fit_transform(
            self.y_train.reshape(-1, 1)).ravel()
        self.y_val = self.scaler_y.transform(self.y_val.reshape(-1, 1)).ravel()
        self.y_test = self.scaler_y.transform(
            self.y_test.reshape(-1, 1)).ravel()

        print(
            f"📊 Train: {self.X_train.shape} | Val: {self.X_val.shape} | Test: {self.X_test.shape}")

        self.train_dataset = TensorDataset(torch.FloatTensor(self.X_train),
                                           torch.FloatTensor(self.y_train))
        self.val_dataset = TensorDataset(torch.FloatTensor(self.X_val),
                                         torch.FloatTensor(self.y_val))
        self.test_dataset = TensorDataset(torch.FloatTensor(self.X_test),
                                          torch.FloatTensor(self.y_test))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class HousingNet(pl.LightningModule):
    """Модель PyTorch Lightning для предсказания цен на жильё"""

    def __init__(self, input_dim=8, hidden_dims=[64, 32, 16], dropout=0.2, lr=0.01):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.network = nn.Sequential(*layers)

        # Буферы для тестового шага (в масштабированных единицах)
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        return self.network(x).squeeze(-1)

    # ====== вспомогательная функция: перевод таргета/предсказаний обратно в исходные единицы ======
    def _inverse_target(self, arr_np: np.ndarray) -> np.ndarray:
        """из масштабированных единиц -> исходные (в десятках тысяч $)"""
        scaler_y = self.trainer.datamodule.scaler_y  # доступ к скейлеру через datamodule
        return scaler_y.inverse_transform(arr_np.reshape(-1, 1)).ravel()

    def _to_usd(self, arr_target_units: np.ndarray) -> np.ndarray:
        """из единиц 'x100k $' -> доллары"""
        return arr_target_units * 100_000.0

    # ================== шаги обучения/валидации/тестирования ==================
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_mae', F.l1_loss(y_hat, y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        val_mae = F.l1_loss(y_hat, y)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', val_mae, on_epoch=True)
        self.log('val_rmse', torch.sqrt(val_loss), on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat, y)
        test_mae = F.l1_loss(y_hat, y)

        self.test_predictions.extend(y_hat.detach().cpu().numpy())
        self.test_targets.extend(y.detach().cpu().numpy())

        self.log('test_loss', test_loss, on_epoch=True)
        self.log('test_mae', test_mae, on_epoch=True)
        self.log('test_rmse', torch.sqrt(test_loss), on_epoch=True)
        return test_loss

    def on_test_epoch_end(self):
        """Итоговые метрики + графики в реальных единицах (долларах)"""
        if not self.test_predictions:
            return

        # 1) из масштабированных -> исходные ('x100k $')
        y_pred_scaled = np.array(self.test_predictions)
        y_true_scaled = np.array(self.test_targets)
        y_pred_orig = self._inverse_target(y_pred_scaled)
        y_true_orig = self._inverse_target(y_true_scaled)

        # 2) в долларах
        y_pred_usd = self._to_usd(y_pred_orig)
        y_true_usd = self._to_usd(y_true_orig)

        # 3) метрики в исходных единицах (x100k $)
        r2 = r2_score(y_true_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        mae = mean_absolute_error(y_true_orig, y_pred_orig)

        # 4) метрики в долларах (удобны для интерпретации)
        rmse_usd = np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))
        mae_usd = mean_absolute_error(y_true_usd, y_pred_usd)

        # Логируем в W&B
        self.log('final_r2', r2)
        self.log('final_rmse_target_units', rmse)  # в x100k $
        self.log('final_mae_target_units', mae)    # в x100k $
        self.log('final_rmse_usd', rmse_usd)
        self.log('final_mae_usd', mae_usd)

        # Диаграмма рассеяния в долларах (реальные значения)
        if self.logger:
            n = min(500, len(y_true_usd))
            table = wandb.Table(
                data=list(zip(y_true_usd[:n], y_pred_usd[:n])),
                columns=["actual_$", "predicted_$"]
            )
            self.logger.experiment.log({
                "Predictions vs Actual (USD)": wandb.plot.scatter(
                    table, "actual_$", "predicted_$",
                    title="Predictions vs Actual (Test) — USD"
                )
            })

            # Несколько примеров
            ex_n = min(50, n)
            examples = wandb.Table(
                columns=["True, $", "Pred, $", "Error, $", "|Error|, $"],
                data=[[float(t), float(p), float(t - p), float(abs(t - p))]
                      for t, p in zip(y_true_usd[:ex_n], y_pred_usd[:ex_n])]
            )
            self.logger.experiment.log({"Prediction examples (USD)": examples})

        print("\n🎯 FINAL RESULTS (in the original target units = x100k $):")
        print(f"📊 R²:   {r2:.4f}")
        print(f"📊 RMSE: {rmse:.4f} x100k $  (~= ${rmse_usd:,.0f})")
        print(f"📊 MAE:  {mae:.4f} x100k $  (~= ${mae_usd:,.0f})")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


def main():
    print("="*70)
    print("🏠 PYTORCH LIGHTNING + W&B - PRICE PREDICTION (real $ on the charts)")
    print("="*70)

    wandb_logger = WandbLogger(
        project="housing-prediction-alex",
        name="pytorch-lightning-iteration1",
        log_model=True
    )

    data_module = HousingDataModule(batch_size=256)

    model = HousingNet(
        input_dim=8,
        hidden_dims=[64, 32, 16],
        dropout=0.2,
        lr=0.01
    )
    print("\n🧠 Model architecture:\n", model)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=True, mode='min')
    checkpoint = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=1, verbose=True)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[early_stopping, checkpoint],
        max_epochs=100,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=10,
        enable_progress_bar=True
    )

    print("\n🚀 Starting training...")
    trainer.fit(model, data_module)

    print("\n📊 Testing the best model...")
    trainer.test(model, data_module, ckpt_path='best')

    print("\n🎉 Training complete!")
    print(f"🌐 W&B dashboard: {wandb_logger.experiment.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
