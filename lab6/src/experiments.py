import os
import subprocess
import yaml
import matplotlib.pyplot as plt

# Folder containing the saved models
saved_model_folder = 'checkpoints/saved_model'

# Path to the test script
test_script = 'test.py'

# List to store model names and corresponding accuracies
model_names = []
accuracies = []

# Iterate through all model files in the saved_model folder
for model_name in sorted(os.listdir(saved_model_folder)):
    model_path = os.path.join(saved_model_folder, model_name)
    if model_path.endswith('.pt'):
        print(f"Running test.py with model: {model_name}")
        
        # Run the test.py script with the current model
        result = subprocess.run(
            ['python', test_script, 
             '--ckpt-path', model_path,
             '--test-batch-size', '4'
             ],
            capture_output=True, text=True
        )
        
        # Extract accuracy from the output
        output = result.stdout
        print("Output:", output)
        for line in output.split('\n'):
            if 'Accuracy:' in line:
                acc = float(line.split('Accuracy: ')[-1])
                accuracies.append(acc)
                model_names.append(model_name)
                print(f"Model: {model_name}, Accuracy: {acc:.4f}")

# Plot the results with a line plot
xlabel = [model_name.split('.')[0].split('_')[-1] for model_name in model_names]
plt.figure(figsize=(20, 10))
plt.plot(xlabel, accuracies, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy over Epochs')

# Save the plot to a file
plt.savefig('accuracy.png')