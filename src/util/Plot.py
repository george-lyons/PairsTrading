import matplotlib.pyplot as plt
import matplotlib.dates as mdates      

def plot_with_highlighted_regions(x_train, y_train, x_test, y_test):
    plt.figure(figsize=(12, 6))

    # Assuming x_train, x_test are index values (like dates) and y_train, y_test are the values to be plotted
    plt.plot(x_train, y_train, label='Training Data')
    plt.plot(x_test, y_test, label='Testing Data')

    # Highlight training region
    plt.axvspan(x_train.index.min(), x_train.index.max(), color='lightblue', alpha=0.3, label='Training Region')

    # Highlight testing region
    plt.axvspan(x_test.index.min(), x_test.index.max(), color='lightgreen', alpha=0.3, label='Testing Region')

    plt.legend()
    plt.title('Trading Strategy with Training and Testing Regions')
    plt.xlabel('Time')
    plt.ylabel('Values')

    # Improve formatting of dates on x-axis (if using dates)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


