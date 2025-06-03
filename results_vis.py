from utils.metrics import MetricsVisualizer
import matplotlib.pyplot as plt

def main():
    # Initialize the metrics visualizer to access performance metrics and charts
    metrics_viz = MetricsVisualizer()
    
    # Print overall accuracy
    accuracy = metrics_viz.performance_metrics.get('accuracy', None)
    if accuracy is not None:
        print(f"Model Accuracy: {accuracy:.2f}")
    
    # Plot and save the accuracy curve
    fig_accuracy = metrics_viz.plot_accuracy_curve()
    fig_accuracy.savefig("accuracy_curve.png")
    print("Saved accuracy curve as 'accuracy_curve.png'.")

    # Plot and save the confusion matrix heatmap
    fig_heatmap = metrics_viz.plot_confusion_matrix()
    fig_heatmap.savefig("confusion_matrix_heatmap.png")
    print("Saved confusion matrix heatmap as 'confusion_matrix_heatmap.png'.")

    # Display other visualizations if needed
    # e.g. loss curve, precision/recall, etc.
    fig_loss = metrics_viz.plot_loss_curve()
    fig_loss.savefig("loss_curve.png")
    print("Saved loss curve as 'loss_curve.png'.")

if __name__ == "__main__":
    main()