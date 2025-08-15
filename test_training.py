#!/usr/bin/env python3
"""
Test script for the CarbonWatch training endpoint
"""

import requests
import json
import time

def test_training_endpoint():
    """Test the /train endpoint"""
    
    print("🚀 Testing CarbonWatch Training Endpoint")
    print("=" * 50)
    
    # Training endpoint URL
    url = "http://localhost:5000/train"
    
    print(f"Sending POST request to {url}")
    print("This may take several minutes...")
    
    start_time = time.time()
    
    try:
        # Send training request
        response = requests.post(url, headers={'Content-Type': 'application/json'})
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️ Training completed in {duration:.2f} seconds")
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Training Successful!")
            print("=" * 30)
            
            # Display training info
            if 'training_info' in result:
                info = result['training_info']
                print(f"📊 Total Samples: {info.get('total_samples', 'N/A')}")
                print(f"🏋️  Training Samples: {info.get('training_samples', 'N/A')}")
                print(f"🧪 Test Samples: {info.get('test_samples', 'N/A')}")
                print(f"📈 Training Epochs: {info.get('training_epochs', 'N/A')}")
                print(f"🔢 Input Dimensions: {info.get('input_dimensions', 'N/A')}")
                print(f"🎯 Target Classes: {info.get('target_classes', 'N/A')}")
            
            # Display performance metrics
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                print(f"\n📊 Performance Metrics:")
                print(f"  Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
                print(f"  Test Loss: {metrics.get('test_loss', 'N/A'):.4f}")
                print(f"  Training Accuracy: {metrics.get('training_accuracy', 'N/A'):.4f}")
                print(f"  Validation Accuracy: {metrics.get('validation_accuracy', 'N/A'):.4f}")
                print(f"  Overfitting Detected: {metrics.get('overfitting_detected', 'N/A')}")
            
            # Display label distribution
            if 'label_distribution' in result:
                print(f"\n📈 Label Distribution:")
                for label, count in result['label_distribution'].items():
                    print(f"  {label}: {count}")
            
            # Display model artifacts info
            if 'model_artifacts' in result:
                artifacts = result['model_artifacts']
                print(f"\n💾 Model Artifacts:")
                print(f"  Storage Mode: {artifacts.get('storage_mode', 'N/A')}")
                
                if 'local_paths' in artifacts:
                    local_paths = artifacts['local_paths']
                    print(f"  📁 Local Paths:")
                    print(f"    Model: {local_paths.get('model_path', 'N/A')}")
                    print(f"    Preprocessor: {local_paths.get('preprocessor_path', 'N/A')}")
                    print(f"    Label Encoder: {local_paths.get('label_encoder_path', 'N/A')}")
                
                if 'cloud_paths' in artifacts and artifacts['cloud_paths']:
                    cloud_paths = artifacts['cloud_paths']
                    print(f"  ☁️  Cloud Paths:")
                    for local_path, cloud_path in cloud_paths.items():
                        print(f"    {local_path} → {cloud_path}")
                elif artifacts.get('storage_mode') == 'cloud':
                    print(f"  ⚠️  Cloud storage configured but no uploads recorded")
                
        else:
            print(f"\n❌ Training Failed!")
            print(f"Status Code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
                print(f"Details: {error_data.get('details', 'No details provided')}")
            except json.JSONDecodeError:
                print(f"Response: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask app is running on localhost:5000")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Training took too long")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def test_prediction_after_training():
    """Test prediction endpoint after training"""
    
    print(f"\n🔮 Testing Prediction After Training")
    print("=" * 40)
    
    url = "http://localhost:5000/predict"
    
    # Sample transaction for testing
    test_transaction = {
        "Transaction Amount": 1500000,
        "Carbon Volume": 30000,
        "Price per Ton": 25,
        "Origin Country": "Indonesia",
        "Cross-Border Flag": 1,
        "Buyer Industry": "Manufacturing",
        "Sudden Transaction Spike": 0,
        "Transaction Hour": 14,
        "Entity Type": "Corporation"
    }
    
    try:
        response = requests.post(url, json=test_transaction)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction Successful!")
            
            if 'predictions' in result:
                pred = result['predictions'][0]
                print(f"🎯 Predicted Label: {pred.get('label', 'N/A')}")
                print(f"🔍 Confidence: {pred.get('confidence', 'N/A'):.4f}")
                print(f"📊 Probabilities: {pred.get('probabilities', 'N/A')}")
        else:
            print(f"❌ Prediction Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Prediction Error: {e}")

if __name__ == "__main__":
    print("🧪 CarbonWatch Training Test")
    print("=" * 50)
    print("Make sure you have:")
    print("1. Supabase configured with transaction data")
    print("2. Environment variables set (.env file)")
    print("3. Flask app running (python app.py)")
    print("=" * 50)
    
    input("Press Enter to start training test...")
    
    # Test training
    test_training_endpoint()
    
    # Test prediction after training
    test_prediction_after_training()
    
    print("\n🎉 Test completed!")
