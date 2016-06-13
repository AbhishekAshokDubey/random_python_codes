import matplotlib.pylab as plt
def save_pediction_plots(Y_train_actual, Y_train_predicted, Y_test_actual, Y_test_predicted, name_terms_array):
#    fig = plt.figure(1)
    plt.subplot(211)
    
    plt.plot(np.asmatrix(range(Y_train_predicted.shape[0])).T, Y_train_predicted, '-k', lw=2, label='train-Predicted')
    plt.plot(np.asmatrix(range(Y_train_predicted.shape[0])).T, Y_train_actual, '-r', lw=2, label='train-actual')
    plt.legend(bbox_to_anchor=(0.9, 0.9), loc=2, borderaxespad=0.)
    
    plt.subplot(212)
    plt.plot(np.asmatrix(range(Y_test_predicted.shape[0])).T, Y_test_predicted, '-k', lw=2, label='test-Predicted')
    plt.plot(np.asmatrix(range(Y_test_predicted.shape[0])).T, Y_test_actual, '-r', lw=2, label='test-actual')
    plt.legend(bbox_to_anchor=(0.9, 0.9), loc=2, borderaxespad=0.)
    #plt.show()
    #plt.close()
    name_terms_str_array = [str(w) for w in name_terms_array]
    plt.savefig("_".join(name_terms_str_array) +".png",dpi=600);
    plt.close()
