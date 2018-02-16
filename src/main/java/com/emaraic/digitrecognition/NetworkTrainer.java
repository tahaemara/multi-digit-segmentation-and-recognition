package com.emaraic.digitrecognition;
import java.io.File;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.deeplearning4j.util.ModelSerializer;
/**
 * I used an example from deeplearning4j to build this example (original autho agibsonccc)
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java
 * @author Taha Emara
 * Website: http://www.emaraic.com
 * Email  : taha@emaraic.com
 * Created on: Dec 18, 2017 
 */
public class NetworkTrainer {
private static final Logger log = LoggerFactory.getLogger(NetworkTrainer.class);

    public static void main(String[] args) throws Exception {
        
        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 64; // Test batch size
        int nEpochs = 5; // Number of training epochs
        int iterations = 1; // Number of training iterations
        int seed = 123; //

       
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        
        log.info("Build model....");

        

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) 
                .regularization(true).l2(0.0005)      
                .learningRate(.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS) 
                //.updater(Updater.ADAM) 
                //.updater(Updater.ADADELTA) 
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, 
                   new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .backprop(true).pretrain(false).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(mnistTrain);
            log.info("*** Completed epoch {} ***", i);
        }
            log.info("Evaluate model....");
            Evaluation eval = model.evaluate(mnistTest);
            log.info(eval.stats());
            //mnistTest.reset();
         log.info("Saving model....");
        ModelSerializer.writeModel(model, new File("cnn-model.data"), true);
    }
    
}
