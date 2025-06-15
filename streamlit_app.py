import streamlit as st
import neuralNetwork as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import mnist

# Configure page
st.set_page_config(
    page_title="🧠 Neural Network Training Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'manager' not in st.session_state:
    st.session_state.manager = nn.ModelManager()

if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

if 'current_losses' not in st.session_state:
    st.session_state.current_losses = []


# Helper functions
@st.cache_data
def load_mnist_data():
    """Load and preprocess MNIST data with caching"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0
    y_train_onehot = nn.one_hot(y_train, 10)
    return (X_train, y_train, y_train_onehot), (X_test, y_test)


def plot_training_loss(losses, model_name):
    """Create a matplotlib plot of training losses"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (Cross-Entropy)")
    ax.set_title(f"Training Loss vs. Epoch - {model_name}")
    ax.grid(True)
    return fig


def plot_digit_sample(image_data, predicted_label, true_label, confidence, model_name, index):
    """Create a matplotlib plot of a digit sample"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_data.reshape(28, 28), cmap="gray")
    ax.set_title(
        f"Model: {model_name} | Index {index}\nPredicted: {predicted_label} | True: {true_label}\nConfidence: {confidence:.2f}")
    ax.axis("off")
    return fig


# Main app
def main():
    st.title("🧠 Neural Network Training & Comparison Tool")
    st.markdown("---")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🎯 Train New Model", "💾 Load Saved Model", "📋 Model Manager",
         "📊 Compare Models", "🔍 Test Individual Model", "💾 Save Model", "🗑️ Remove Model"]
    )

    # Display current models in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Models")
    if st.session_state.manager.models:
        for name, info in st.session_state.manager.model_info.items():
            accuracy = info.get('test_accuracy', 'Unknown')
            if isinstance(accuracy, float):
                accuracy = f"{accuracy * 100:.2f}%"
            st.sidebar.write(f"🤖 **{name}**: {accuracy}")
    else:
        st.sidebar.write("No models loaded")

    # Page routing
    if page == "🎯 Train New Model":
        train_new_model_page()
    elif page == "💾 Load Saved Model":
        load_saved_model_page()
    elif page == "📋 Model Manager":
        model_manager_page()
    elif page == "📊 Compare Models":
        compare_models_page()
    elif page == "🔍 Test Individual Model":
        test_individual_model_page()
    elif page == "💾 Save Model":
        save_model_page()
    elif page == "🗑️ Remove Model":
        remove_model_page()


def train_new_model_page():
    st.header("🎯 Train New Model")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ Training Parameters")

        # Model name
        model_name = st.text_input("Model Name", value=f"model_{len(st.session_state.manager.models) + 1}")

        # Hyperparameters
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.1, step=0.01,
                                        format="%.4f")
        epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=20, step=1)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=64, step=1)

        # L2 Regularization
        use_l2 = st.checkbox("Use L2 Regularization")
        l2_lambda = 0.0
        if use_l2:
            l2_lambda = st.number_input("L2 Regularization Coefficient", min_value=0.0, max_value=1.0, value=0.001,
                                        step=0.001, format="%.4f")

        verbose = st.checkbox("Verbose Training", value=True)

    with col2:
        st.subheader("🏗️ Network Architecture")

        # Architecture options
        architecture_choice = st.selectbox(
            "Choose Architecture",
            ["Default ([784, 128, 64, 10])", "Custom"]
        )

        if architecture_choice == "Default ([784, 128, 64, 10])":
            layers = [784, 128, 64, 10]
            st.write("**Selected Architecture:**", layers)
        else:
            num_hidden = st.number_input("Number of Hidden Layers", min_value=1, max_value=7, value=2, step=1)
            layers = [784]  # Input layer

            for i in range(num_hidden):
                layer_size = st.number_input(f"Hidden Layer {i + 1} Size", min_value=1, max_value=1000, value=128,
                                             step=1, key=f"layer_{i}")
                layers.append(layer_size)

            layers.append(10)  # Output layer
            st.write("**Selected Architecture:**", layers)

    # Training button
    if st.button("🏋️ Start Training", type="primary"):
        if model_name.strip():
            with st.spinner("Loading MNIST data..."):
                train_data, test_data = load_mnist_data()
                X_train, y_train, y_train_onehot = train_data
                X_test, y_test = test_data

            # Create settings object
            settings = nn.Settings(learning_rate, epochs, batch_size, verbose, l2_lambda)

            # Create progress bars
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_container = st.empty()

            # Train model
            with st.spinner(f"Training model '{model_name}'..."):
                net = nn.OurNeuralNetwork(layers, learning_rate=learning_rate, l2_lambda=l2_lambda)

                # Custom training loop with progress updates
                n = len(X_train)
                losses = []

                for epoch in range(1, epochs + 1):
                    # Shuffle data
                    perm = np.random.permutation(n)
                    X_shuffled, Y_shuffled = X_train[perm], y_train_onehot[perm]

                    epoch_loss = 0.0

                    # Mini-batch training
                    for i in range(0, n, batch_size):
                        x_batch = X_shuffled[i: i + batch_size]
                        y_batch = Y_shuffled[i: i + batch_size]
                        batch_loss = net.train(x_batch, y_batch)
                        epoch_loss += batch_loss * len(x_batch)

                    epoch_loss /= n
                    losses.append(epoch_loss)

                    # Update progress
                    progress = epoch / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

                    # Update loss plot every 5 epochs or on final epoch
                    if epoch % 5 == 0 or epoch == epochs:
                        with loss_container.container():
                            fig = plot_training_loss(losses, model_name)
                            st.pyplot(fig)
                            plt.close(fig)

            # Evaluate model
            with st.spinner("Evaluating model..."):
                activates, _ = net.feedforward(X_test)
                y_pred_probs = activates[-1]
                y_pred_labels = np.argmax(y_pred_probs, axis=1)
                accuracy = np.sum(y_pred_labels == y_test) / len(y_test)

            # Add to manager
            st.session_state.manager.add_model(model_name, net, settings, accuracy, losses)

            # Display results
            st.success(f"✅ Training complete! Test accuracy: {accuracy * 100:.2f}%")

        else:
            st.error("Please enter a valid model name")


def load_saved_model_page():
    st.header("💾 Load Saved Model")

    # List available models
    model_files = nn.list_saved_models()

    if model_files:
        st.subheader("Available Saved Models")
        for i, filename in enumerate(model_files, 1):
            st.write(f"{i}. {filename}")

        # File selection
        selected_file = st.selectbox("Select a model file:", model_files)

        # Model name input
        default_name = selected_file.replace('.pkl', '') if selected_file else ""
        model_name = st.text_input("Name for loaded model:", value=default_name)

        if st.button("Load Model"):
            if selected_file and model_name:
                if st.session_state.manager.load_model(selected_file, model_name):
                    st.success(f"Model '{model_name}' loaded successfully!")
                    st.rerun()
            else:
                st.error("Please select a file and enter a model name")
    else:
        st.info("No saved model files found in the current directory.")


def model_manager_page():
    st.header("📋 Model Manager")

    if not st.session_state.manager.models:
        st.info("No models currently loaded in the manager.")
        return

    st.subheader("Loaded Models")

    # Create a table of models
    model_data = []
    for name, info in st.session_state.manager.model_info.items():
        accuracy = info.get('test_accuracy', 'Unknown')
        final_loss = info.get('final_loss', 'Unknown')
        epochs = info.get('total_epochs', 'Unknown')

        if isinstance(accuracy, float):
            accuracy = f"{accuracy * 100:.2f}%"
        if isinstance(final_loss, float):
            final_loss = f"{final_loss:.4f}"

        model_data.append({
            'Model Name': name,
            'Accuracy': accuracy,
            'Final Loss': final_loss,
            'Epochs': epochs
        })

    df = pd.DataFrame(model_data)
    st.dataframe(df, use_container_width=True)

    # Detailed view
    st.subheader("Detailed Model Information")
    selected_model = st.selectbox("Select a model for details:", list(st.session_state.manager.models.keys()))

    if selected_model:
        info = st.session_state.manager.model_info[selected_model]
        model = st.session_state.manager.models[selected_model]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Test Accuracy",
                      f"{info.get('test_accuracy', 0) * 100:.2f}%" if isinstance(info.get('test_accuracy'),
                                                                                 float) else "Unknown")
            st.metric("Final Loss",
                      f"{info.get('final_loss', 0):.4f}" if isinstance(info.get('final_loss'), float) else "Unknown")

        with col2:
            st.metric("Total Epochs", info.get('total_epochs', 'Unknown'))

        # Network Architecture
        st.subheader("Network Architecture")
        if hasattr(model, 'weights') and model.weights:
            # Extract and display layer information
            layer_info = []
            layer_sizes = [model.weights[0].shape[1]]  # Input layer

            for i, weight in enumerate(model.weights):
                layer_sizes.append(weight.shape[0])
                layer_info.append({
                    'Layer': f"Layer {i + 1}",
                    'Type': 'Hidden (ReLU)' if i < len(model.weights) - 1 else 'Output (Softmax)',
                    'Input Size': weight.shape[1],
                    'Output Size': weight.shape[0],
                    'Parameters': weight.shape[0] * weight.shape[1] + weight.shape[0]  # weights + biases
                })

            # Display architecture summary
            st.write(f"**Architecture:** {layer_sizes}")

            # Display detailed layer information
            layer_df = pd.DataFrame(layer_info)
            st.dataframe(layer_df, use_container_width=True)

            # Calculate total parameters
            total_params = sum([info['Parameters'] for info in layer_info])
            st.metric("Total Parameters", f"{total_params:,}")

        else:
            st.write("Architecture information not available")

        # Settings
        if 'settings' in info and info['settings']:
            settings = info['settings']
            st.subheader("Training Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Learning Rate:** {settings.learning_rate}")
            with col2:
                st.write(f"**Batch Size:** {settings.batch_size}")
            with col3:
                st.write(f"**L2 Lambda:** {settings.l2_lambda}")


def compare_models_page():
    st.header("📊 Compare Models")

    if len(st.session_state.manager.models) < 2:
        st.warning("Need at least 2 models to compare. Please train or load more models.")
        return

    # Model selection
    all_models = list(st.session_state.manager.models.keys())
    selected_models = st.multiselect("Select models to compare (leave empty for all):", all_models)

    if not selected_models:
        selected_models = all_models

    if st.button("Compare Selected Models"):
        # Prepare comparison data
        comparison_data = []
        for name in selected_models:
            if name in st.session_state.manager.models:
                info = st.session_state.manager.model_info[name]
                accuracy = info.get('test_accuracy', 0)
                final_loss = info.get('final_loss', float('inf'))
                epochs = info.get('total_epochs', 0)
                settings = info.get('settings')

                comparison_data.append({
                    'Model Name': name,
                    'Accuracy': accuracy * 100 if isinstance(accuracy, float) else 0,
                    'Final Loss': final_loss if isinstance(final_loss, float) else float('inf'),
                    'Epochs': epochs,
                    'Learning Rate': settings.learning_rate if settings else 'Unknown',
                    'Batch Size': settings.batch_size if settings else 'Unknown',
                    'L2 Lambda': settings.l2_lambda if settings else 'Unknown'
                })

        # Sort by accuracy
        comparison_data.sort(key=lambda x: x['Accuracy'], reverse=True)

        # Display comparison
        st.subheader("Model Comparison Results")

        # Ranking
        for i, model in enumerate(comparison_data, 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."

            with st.expander(f"{rank_emoji} {model['Model Name']} - {model['Accuracy']:.2f}%"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{model['Accuracy']:.2f}%")
                    st.metric("Final Loss",
                              f"{model['Final Loss']:.4f}" if model['Final Loss'] != float('inf') else "Unknown")
                with col2:
                    st.metric("Epochs", model['Epochs'])
                    st.metric("Learning Rate", model['Learning Rate'])
                with col3:
                    st.metric("Batch Size", model['Batch Size'])
                    st.metric("L2 Lambda", model['L2 Lambda'])

        # Comparison chart
        st.subheader("Accuracy Comparison")
        df_chart = pd.DataFrame(comparison_data)
        st.bar_chart(df_chart.set_index('Model Name')['Accuracy'])


def test_individual_model_page():
    st.header("🔍 Test Individual Model")

    if not st.session_state.manager.models:
        st.warning("No models available. Please train or load a model first.")
        return

    # Model selection
    selected_model = st.selectbox("Select a model to test:", list(st.session_state.manager.models.keys()))

    if selected_model:
        model = st.session_state.manager.get_model(selected_model)

        # Load test data
        _, (X_test, y_test) = load_mnist_data()

        # Number of samples
        num_samples = st.number_input("Number of random samples to test:", min_value=1, max_value=100, value=5)

        if st.button("Generate Random Tests"):
            test_indices = np.random.randint(0, len(X_test), size=num_samples)

            st.subheader("Test Results")

            for idx in test_indices:
                x = X_test[idx]
                true_label = y_test[idx]

                # Make prediction
                activs, _ = model.feedforward(x[np.newaxis, :])
                probs = activs[-1]
                pred_label = np.argmax(probs, axis=1)[0]
                confidence = probs[0, pred_label]

                # Create two columns
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Display image
                    fig = plot_digit_sample(x, pred_label, true_label, confidence, selected_model, idx)
                    st.pyplot(fig)
                    plt.close(fig)

                with col2:
                    # Display prediction details
                    st.write(f"**Sample Index:** {idx}")
                    st.write(f"**True Label:** {true_label}")
                    st.write(f"**Predicted Label:** {pred_label}")
                    st.write(f"**Confidence:** {confidence:.4f}")

                    # Show all probabilities
                    st.write("**All Class Probabilities:**")
                    prob_data = pd.DataFrame({
                        'Digit': range(10),
                        'Probability': probs[0]
                    })
                    st.bar_chart(prob_data.set_index('Digit'))

                    # Result
                    if pred_label == true_label:
                        st.success("✅ Correct Prediction!")
                    else:
                        st.error("❌ Incorrect Prediction")

                st.markdown("---")


def save_model_page():
    st.header("💾 Save Model")

    if not st.session_state.manager.models:
        st.warning("No models to save. Please train or load a model first.")
        return

    # Model selection
    selected_model = st.selectbox("Select a model to save:", list(st.session_state.manager.models.keys()))

    if selected_model:
        # Filename input
        default_filename = f"{selected_model}.pkl"
        filename = st.text_input("Filename:", value=default_filename)

        if st.button("Save Model"):
            if filename:
                if not filename.endswith('.pkl'):
                    filename += '.pkl'

                if st.session_state.manager.save_model(selected_model, filename):
                    st.success(f"Model '{selected_model}' saved as '{filename}'")
            else:
                st.error("Please enter a filename")


def remove_model_page():
    st.header("🗑️ Remove Model")

    if not st.session_state.manager.models:
        st.warning("No models to remove.")
        return

    # Model selection
    selected_model = st.selectbox("Select a model to remove:", list(st.session_state.manager.models.keys()))

    if selected_model:
        st.warning(f"Are you sure you want to remove '{selected_model}'? This action cannot be undone.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Yes, Remove Model", type="primary"):
                st.session_state.manager.remove_model(selected_model)
                st.success(f"Model '{selected_model}' removed successfully!")
                st.rerun()

        with col2:
            if st.button("Cancel"):
                st.info("Operation cancelled.")


if __name__ == "__main__":
    main()