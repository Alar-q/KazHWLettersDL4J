package com.rat6;

import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Random;

public class LeNet {
    /*
    private static final Logger log = LoggerFactory.getLogger(DataTrain.class);
    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();
        new Nd4j();
        double learningRate = 0.05; // Число входных каналов
        int nChannels = 1; // Число входных каналов

        int outputNum = 42; // Число возможных исходов
        int batchSize = 64; // Размер тестового пакета
        int nEpochs = 10; // Число периодов обучения
        int seed = 123;
        int WIDTH = 32;

        int[] inputShape = new int[] {nChannels, WIDTH, WIDTH};

        log.info("Загружаются данные....");
        String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        Random rand = new Random();
        File root = new File("C:\\Users\\Pupochek\\Documents\\kazLetRedacted");
        FileSplit f = new FileSplit(root, allowedExtensions, rand);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader reader;
        DataSetIterator data;
        reader = new ImageRecordReader(WIDTH, WIDTH, 1, labelMaker);
        reader.initialize(f);

        data = new RecordReaderDataSetIterator(reader, batchSize, 1, outputNum);


        log.info("Построение модели....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) // Задается число итераций
                .regularization(true).l2(0.0005)
                /*
                Раскомментировать следующие строки, чтобы задать затухание
                и смещение скорости обучения

                //.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse)
                //.lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
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
                        // Заметьте, что в последующих слоях задавать nIn не нужно
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
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction
                        .NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .backprop(true).pretrain(false).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Обучение модели....");
        model.setListeners(new ScoreIterationListener(10));

        for (int i = 0; i < nEpochs; i++) {
            System.out.println("epoch: " + i);
            model.fit(data);
        }



        allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        root = new File("C:\\Users\\Pupochek\\Documents\\kazLetRedacted");
        f = new FileSplit(root, allowedExtensions, rand);
        labelMaker = new ParentPathLabelGenerator();

        reader = new ImageRecordReader(WIDTH, WIDTH, 1, labelMaker);
        reader.initialize(f);
        data = new RecordReaderDataSetIterator(reader, batchSize, 1, outputNum);

        Evaluation eval = new Evaluation(outputNum);
        while (data.hasNext()) {
            DataSet ds = data.next();
            INDArray output = model.output(ds.getFeatures(), false);
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
        data.reset();

        model.save(new File("C:\\weights\\kazlettersLeNet.zip"));

        log.info("****************Конец обучения********************");
    }
    */
}
