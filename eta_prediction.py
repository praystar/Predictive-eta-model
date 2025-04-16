import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

class ETAPredictor:
    def __init__(self):
        self.model = None
        self.cost_model = None
        self.feature_columns = None
        self.target_column = None
        self.cost_column = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def plot_learning_curves(self, X, y, model_type='eta'):
        """Plot learning curves to visualize overfitting"""
        print(f"\nGenerating learning curves for {model_type} model...")
        
        # Create XGBoost model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100
        )
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        
        # Calculate mean and standard deviation
        train_mean = np.sqrt(-train_scores.mean(axis=1))
        train_std = np.sqrt(train_scores.std(axis=1))
        test_mean = np.sqrt(-test_scores.mean(axis=1))
        test_std = np.sqrt(test_scores.std(axis=1))
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score', color='blue')
        plt.plot(train_sizes, test_mean, label='Cross-validation score', color='red')
        
        # Plot the bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
        
        # Add labels and title
        plt.xlabel('Number of training examples')
        plt.ylabel('RMSE')
        plt.title(f'Learning Curves for {model_type.upper()} Model')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f'plots/{model_type}_learning_curves.png')
        plt.close()
        
        print(f"✓ Saved learning curves to plots/{model_type}_learning_curves.png")
        
        # Print gap between training and validation scores
        final_gap = abs(train_mean[-1] - test_mean[-1])
        print(f"Final gap between training and validation scores: {final_gap:.2f}")
        if final_gap > 0.1:
            print("Warning: Large gap detected - model might be overfitting!")
        else:
            print("Good: Small gap between training and validation scores")
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the data"""
        print("\n" + "="*80)
        print("Loading and preprocessing data...")
        print("="*80)
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found at: {file_path}")
            
            # Try to read the file
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading CSV file: {str(e)}")
                print("\nTrying to read with different encodings...")
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"Successfully read file with {encoding} encoding")
                        break
                    except:
                        continue
                else:
                    raise ValueError("Could not read file with any standard encoding")
            
            print("\nAvailable columns in dataset:")
            print(tabulate([[col] for col in df.columns], headers=['Column Name'], tablefmt='grid'))
            
            # Print first few rows to verify data
            print("\nFirst few rows of data:")
            print(tabulate(df.head(), headers='keys', tablefmt='grid'))
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        
        # Basic preprocessing
        print(f"\nInitial shape: {df.shape}")
        df = df.dropna()
        df = df.drop_duplicates()
        print(f"Shape after cleaning: {df.shape}")
        
        # Convert date columns to datetime with error handling
        date_columns = [
            'scheduled delivery date',
            'delivered to client date',
            'delivery recorded date',
            'scheduled_date',
            'delivery_date',
            'recorded_date',
            'shipment_date',
            'arrival_date'
        ]
        
        print("\nProcessing date columns:")
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    print(f"✓ Successfully converted {col} to datetime")
                except Exception as e:
                    print(f"✗ Could not convert {col} to datetime: {str(e)}")
                    if col in df.columns:
                        df = df.drop(col, axis=1)
        
        # Calculate ETA using available date columns
        eta_calculated = False
        date_pairs = [
            ('delivered to client date', 'scheduled delivery date'),
            ('delivery recorded date', 'scheduled delivery date'),
            ('delivery_date', 'scheduled_date'),
            ('arrival_date', 'shipment_date')
        ]
        
        for end_col, start_col in date_pairs:
            if end_col in df.columns and start_col in df.columns:
                try:
                    df['eta'] = (df[end_col] - df[start_col]).dt.total_seconds() / 3600
                    print(f"\n✓ Calculated ETA using {end_col} and {start_col}")
                    eta_calculated = True
                    break
                except Exception as e:
                    print(f"✗ Could not calculate ETA using {end_col} and {start_col}: {str(e)}")
        
        if not eta_calculated:
            # Try to find existing ETA or duration column
            eta_columns = [col for col in df.columns if 'eta' in col.lower() or 'duration' in col.lower() or 'time' in col.lower()]
            for col in eta_columns:
                try:
                    df['eta'] = pd.to_numeric(df[col], errors='coerce')
                    print(f"\n✓ Using existing {col} as ETA")
                    eta_calculated = True
                    break
                except:
                    continue
        
        if not eta_calculated:
            raise ValueError("Could not calculate or find ETA in the dataset")
        
        # Handle freight cost
        cost_columns = [
            'freight cost', 'freight_cost', 'shipping_cost', 'transport_cost',
            'cost', 'price', 'amount', 'charge'
        ]
        
        cost_found = False
        for col in cost_columns:
            if col in df.columns:
                df['freight_cost'] = pd.to_numeric(df[col], errors='coerce')
                print(f"\n✓ Processed freight cost from column: {col}")
                cost_found = True
                break
        
        if not cost_found:
            print("\n✗ No freight cost column found")
        
        # Handle additional features
        feature_categories = {
            'speed': ['speed', 'velocity', 'avg_speed'],
            'distance': ['distance', 'route_length', 'total_distance'],
            'traffic': ['traffic', 'traffic_level', 'congestion'],
            'weather': ['weather', 'weather_condition', 'temperature', 'rainfall']
        }
        
        print("\nProcessing additional features:")
        for category, variations in feature_categories.items():
            for var in variations:
                if var in df.columns:
                    try:
                        df[category] = pd.to_numeric(df[var], errors='coerce')
                        print(f"✓ Added {category} feature from {var}")
                        break
                    except:
                        continue
        
        # Handle categorical columns
        print("\nProcessing categorical columns...")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in date_columns:
                df[col] = self.label_encoder.fit_transform(df[col].astype(str))
                print(f"✓ Converted {col} to categorical codes")
        
        # Select features for prediction
        self.feature_columns = [
            col for col in df.columns 
            if col not in ['eta', 'freight_cost'] 
            and not col.endswith('_time') 
            and not col.endswith('_date')
            and col not in date_columns
        ]
        
        print("\nSelected features for prediction:")
        print(tabulate([[col] for col in self.feature_columns], headers=['Feature'], tablefmt='grid'))
        
        self.target_column = 'eta'
        self.cost_column = 'freight_cost' if 'freight_cost' in df.columns else None
        
        # Prepare features and targets
        X = df[self.feature_columns]
        y_eta = df[self.target_column]
        y_cost = df[self.cost_column] if self.cost_column is not None else None
        
        # Remove any remaining NaN values
        X = X.fillna(X.mean())
        y_eta = y_eta.fillna(y_eta.mean())
        if y_cost is not None:
            y_cost = y_cost.fillna(y_cost.mean())
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        print(f"\nFinal feature matrix shape: {X.shape}")
        print(f"ETA target vector shape: {y_eta.shape}")
        if y_cost is not None:
            print(f"Cost target vector shape: {y_cost.shape}")
        
        return X, y_eta, y_cost
    
    def train_models(self, X, y_eta, y_cost):
        """Train XGBoost models for ETA and cost prediction"""
        print("\n" + "="*80)
        print("Training models...")
        print("="*80)
        
        # Split data into training (80%) and testing (20%) sets
        X_train, X_test, y_eta_train, y_eta_test = train_test_split(
            X, y_eta, test_size=0.2, random_state=42
        )
        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Plot learning curves for ETA model
        self.plot_learning_curves(X_train, y_eta_train, 'eta')
        
        # Perform cross-validation for ETA model
        print("\nPerforming cross-validation for ETA model...")
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100
            ),
            X_train, y_eta_train,
            cv=kfold,
            scoring='neg_mean_squared_error'
        )
        print(f"Cross-validation RMSE scores: {np.sqrt(-cv_scores)}")
        print(f"Mean CV RMSE: {np.sqrt(-cv_scores.mean()):.2f} ± {np.sqrt(cv_scores.std()):.2f}")
        
        # Train ETA model
        print("\nTraining ETA prediction model...")
        dtrain_eta = xgb.DMatrix(X_train, label=y_eta_train)
        dtest_eta = xgb.DMatrix(X_test, label=y_eta_test)
        
        params_eta = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100
        }
        
        self.model = xgb.train(
            params_eta,
            dtrain_eta,
            num_boost_round=100,
            evals=[(dtrain_eta, 'train'), (dtest_eta, 'test')],
            early_stopping_rounds=10,
            verbose_eval=True
        )
        
        # Train cost model if cost data is available
        if y_cost is not None:
            print("\nTraining cost prediction model...")
            y_cost_train = y_cost.iloc[X_train.index]
            y_cost_test = y_cost.iloc[X_test.index]
            
            # Plot learning curves for cost model
            self.plot_learning_curves(X_train, y_cost_train, 'cost')
            
            # Perform cross-validation for cost model
            print("\nPerforming cross-validation for cost model...")
            cv_scores_cost = cross_val_score(
                xgb.XGBRegressor(
                    objective='reg:squarederror',
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100
                ),
                X_train, y_cost_train,
                cv=kfold,
                scoring='neg_mean_squared_error'
            )
            print(f"Cross-validation RMSE scores: {np.sqrt(-cv_scores_cost)}")
            print(f"Mean CV RMSE: {np.sqrt(-cv_scores_cost.mean()):.2f} ± {np.sqrt(cv_scores_cost.std()):.2f}")
            
            dtrain_cost = xgb.DMatrix(X_train, label=y_cost_train)
            dtest_cost = xgb.DMatrix(X_test, label=y_cost_test)
            
            params_cost = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
            
            self.cost_model = xgb.train(
                params_cost,
                dtrain_cost,
                num_boost_round=100,
                evals=[(dtrain_cost, 'train'), (dtest_cost, 'test')],
                early_stopping_rounds=10,
                verbose_eval=True
            )
        
        return X_test, y_eta_test, y_cost_test if y_cost is not None else None
    
    def evaluate_models(self, X_test, y_eta_test, y_cost_test):
        """Evaluate model performance"""
        print("\n" + "="*80)
        print("Evaluating models...")
        print("="*80)
        
        # Make ETA predictions
        dtest = xgb.DMatrix(X_test)
        y_eta_pred = self.model.predict(dtest)
        
        # Calculate ETA metrics
        mae_eta = mean_absolute_error(y_eta_test, y_eta_pred)
        rmse_eta = np.sqrt(mean_squared_error(y_eta_test, y_eta_pred))
        r2_eta = r2_score(y_eta_test, y_eta_pred)
        
        # Calculate F-score for ETA (using bins with duplicates handling)
        try:
            eta_bins = pd.qcut(y_eta_test, q=5, labels=False, duplicates='drop')
            pred_bins = pd.qcut(y_eta_pred, q=5, labels=False, duplicates='drop')
            f1_eta = f1_score(eta_bins, pred_bins, average='weighted')
        except Exception as e:
            print(f"Warning: Could not calculate F1 score for ETA: {str(e)}")
            print("Using alternative binning method...")
            # Use fixed-width bins instead
            eta_bins = pd.cut(y_eta_test, bins=5, labels=False)
            pred_bins = pd.cut(y_eta_pred, bins=5, labels=False)
            f1_eta = f1_score(eta_bins, pred_bins, average='weighted')
        
        print("\nETA Prediction Performance:")
        print(tabulate([
            ['MAE', f'{mae_eta:.2f} hours'],
            ['RMSE', f'{rmse_eta:.2f} hours'],
            ['R2 Score', f'{r2_eta:.2f}'],
            ['F1 Score', f'{f1_eta:.2f}']
        ], headers=['Metric', 'Value'], tablefmt='grid'))
        
        # Make cost predictions if cost model exists
        if self.cost_model is not None and y_cost_test is not None:
            y_cost_pred = self.cost_model.predict(dtest)
            
            # Calculate cost metrics
            mae_cost = mean_absolute_error(y_cost_test, y_cost_pred)
            rmse_cost = np.sqrt(mean_squared_error(y_cost_test, y_cost_pred))
            r2_cost = r2_score(y_cost_test, y_cost_pred)
            
            # Calculate F-score for cost (using bins with duplicates handling)
            try:
                cost_bins = pd.qcut(y_cost_test, q=5, labels=False, duplicates='drop')
                pred_cost_bins = pd.qcut(y_cost_pred, q=5, labels=False, duplicates='drop')
                f1_cost = f1_score(cost_bins, pred_cost_bins, average='weighted')
            except Exception as e:
                print(f"Warning: Could not calculate F1 score for cost: {str(e)}")
                print("Using alternative binning method...")
                # Use fixed-width bins instead
                cost_bins = pd.cut(y_cost_test, bins=5, labels=False)
                pred_cost_bins = pd.cut(y_cost_pred, bins=5, labels=False)
                f1_cost = f1_score(cost_bins, pred_cost_bins, average='weighted')
            
            print("\nCost Prediction Performance:")
            print(tabulate([
                ['MAE', f'${mae_cost:.2f}'],
                ['RMSE', f'${rmse_cost:.2f}'],
                ['R2 Score', f'{r2_cost:.2f}'],
                ['F1 Score', f'{f1_cost:.2f}']
            ], headers=['Metric', 'Value'], tablefmt='grid'))
            
            # Save combined predictions
            results = pd.DataFrame({
                'Actual_ETA': y_eta_test,
                'Predicted_ETA': y_eta_pred,
                'ETA_Error': y_eta_test - y_eta_pred,
                'Actual_Cost': y_cost_test,
                'Predicted_Cost': y_cost_pred,
                'Cost_Error': y_cost_test - y_cost_pred
            })
        else:
            # Save ETA predictions only
            results = pd.DataFrame({
                'Actual_ETA': y_eta_test,
                'Predicted_ETA': y_eta_pred,
                'ETA_Error': y_eta_test - y_eta_pred
            })
        
        results.to_csv('predictions.csv', index=False)
        print("\n✓ Saved predictions to predictions.csv")
        
        return y_eta_pred, y_cost_pred if self.cost_model is not None else None
    
    def plot_results(self, y_eta_test, y_eta_pred, y_cost_test=None, y_cost_pred=None):
        """Create visualization plots"""
        print("\n" + "="*80)
        print("Creating visualizations...")
        print("="*80)
        
        # Create output directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # ETA plots
        plt.figure(figsize=(10, 6))
        plt.scatter(y_eta_test, y_eta_pred, alpha=0.5)
        plt.plot([y_eta_test.min(), y_eta_test.max()], [y_eta_test.min(), y_eta_test.max()], 'r--')
        plt.xlabel('Actual ETA (hours)')
        plt.ylabel('Predicted ETA (hours)')
        plt.title('Actual vs Predicted ETA')
        plt.savefig('plots/eta_prediction.png')
        plt.close()
        
        # Cost plots if available
        if y_cost_test is not None and y_cost_pred is not None:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_cost_test, y_cost_pred, alpha=0.5)
            plt.plot([y_cost_test.min(), y_cost_test.max()], [y_cost_test.min(), y_cost_test.max()], 'r--')
            plt.xlabel('Actual Cost ($)')
            plt.ylabel('Predicted Cost ($)')
            plt.title('Actual vs Predicted Cost')
            plt.savefig('plots/cost_prediction.png')
            plt.close()
        
        # Feature importance plots
        plt.figure(figsize=(12, 6))
        xgb.plot_importance(self.model, max_num_features=20)
        plt.title('Top 20 Most Important Features for ETA Prediction')
        plt.tight_layout()
        plt.savefig('plots/eta_feature_importance.png')
        plt.close()
        
        if self.cost_model is not None:
            plt.figure(figsize=(12, 6))
            xgb.plot_importance(self.cost_model, max_num_features=20)
            plt.title('Top 20 Most Important Features for Cost Prediction')
            plt.tight_layout()
            plt.savefig('plots/cost_feature_importance.png')
            plt.close()
        
        print("\n✓ Saved visualization plots to 'plots' directory")

def main():
    try:
        # Initialize predictor
        predictor = ETAPredictor()
        
        # Specify the path to your CSV file
        # You can change this path to point to your desired CSV file
        data_file = r'E:\Dataset1.csv'  # Change this path to your CSV file
        
        print(f"\nUsing data file: {data_file}")
        
        # Load and preprocess data
        X, y_eta, y_cost = predictor.load_and_preprocess_data(data_file)
        
        # Train models
        X_test, y_eta_test, y_cost_test = predictor.train_models(X, y_eta, y_cost)
        
        # Evaluate models
        y_eta_pred, y_cost_pred = predictor.evaluate_models(X_test, y_eta_test, y_cost_test)
        
        # Create visualizations
        predictor.plot_results(y_eta_test, y_eta_pred, y_cost_test, y_cost_pred)
        
        print("\n" + "="*80)
        print("Process completed successfully!")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("Error:", str(e))
        print("\nPlease check the error message above and ensure:")
        print("1. The CSV file path is correct")
        print("2. The file contains the required columns:")
        print("   - Date columns (various names supported)")
        print("   - Freight cost (optional)")
        print("   - Additional features (speed, distance, traffic, weather)")
        print("3. The data format is correct")
        print("="*80)
        raise

if __name__ == "__main__":
    main() 