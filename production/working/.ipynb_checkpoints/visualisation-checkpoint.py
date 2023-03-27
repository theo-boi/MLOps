import json
import matplotlib.pyplot as plt


def plot(predictions_data, last=0, hspace=1, fig_size=(16, 16)):
    if last <= 0:
        sample = predictions_data
    sample = predictions_data[-last:]

    cols = int(len(sample)**.5)
    rows = (len(sample) + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=fig_size)
    fig.subplots_adjust(hspace=hspace)

    def ax(i,j):
        if rows > 1 and cols > 1:
            return axs[i,j]
        elif rows > 1:
            return axs[i]
        elif cols > 1:
            return axs[j]
        else:
            return axs
    
    for i in range(rows):
        for j in range(cols):
            if i*cols+j >= len(sample):
                ax(i,j).axis('off')
            else:
                pred, data = sample[i*cols+j]
                ax(i,j).imshow(data.reshape(28,28), cmap='gray')
                ax(i,j).set_title(json.loads(pred)['predictions'])
                ax(i,j).axis('off')

    plt.show()
