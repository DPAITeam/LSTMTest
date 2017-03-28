package com.zsh_o.dl;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;

/**
 * Created by zsh_o on 2017/3/23.
 */
public class JSTMTest {

    public static void main(String[] args) throws Exception {

        String basePath=new File(JSTMTest.class.getResource("/").getFile()).getPath()+"/";

        int miniBatchSize = 32;

        int numInput=17;
        int numOutput=1;
        int nEpochs = 100;
        int iterations=2;
        int hiddenSize=17;
        String limiter="\t";
        boolean isread=true;
        boolean saveUpdater = true;

        String savedPath=basePath+"LSTM_Model.zip";
        File locationToSave = new File(savedPath);

        // ----- Load the training data -----
        SequenceRecordReader trainFeatures=new CSVSequenceRecordReader(0,limiter);
        trainFeatures.initialize(new NumberedFileInputSplit(basePath+"p_train_features_%d.csv",0,0));
        SequenceRecordReader trainLabels=new CSVSequenceRecordReader(0,limiter);
        trainLabels.initialize(new NumberedFileInputSplit(basePath+"p_train_labels_%d.csv",0,0));
        DataSetIterator trainDataIter=new SequenceRecordReaderDataSetIterator(trainFeatures,trainLabels,miniBatchSize,-1,true, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);
        DataSet trainData=trainDataIter.next();

        SequenceRecordReader testFeatures=new CSVSequenceRecordReader(0,limiter);
        testFeatures.initialize(new NumberedFileInputSplit(basePath+"p_test_features_%d.csv",0,0));
        SequenceRecordReader testLabels=new CSVSequenceRecordReader(0,limiter);
        testLabels.initialize(new NumberedFileInputSplit(basePath+"p_test_labels_%d.csv",0,0));
        DataSetIterator testDataIter=new SequenceRecordReaderDataSetIterator(testFeatures,testLabels,miniBatchSize,-1,true, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);
        DataSet testData=testDataIter.next();

        //Normalize data, including labels (fitLabel=true)
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainData);              //Collect training data statistics

        normalizer.transform(trainData);
        normalizer.transform(testData);

        MultiLayerNetwork net=null;


        if(!isread){
            // ----- Configure the network -----
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(iterations)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(0.0015)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInput).nOut(hiddenSize)
                    .build())
                .layer(1, new GravesLSTM.Builder().activation(Activation.TANH).nIn(hiddenSize).nOut(hiddenSize)
                    .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.IDENTITY).nIn(hiddenSize).nOut(numOutput).build())
                .build();

            net = new MultiLayerNetwork(conf);
            net.init();

            UIServer uiServer = UIServer.getInstance();
            uiServer.enableRemoteListener();
            StatsStorageRouter remoteUIRouter = new RemoteUIStatsStorageRouter("http://localhost:9000");
            net.setListeners(new StatsListener(remoteUIRouter));

            // ----- Train the network, evaluating the test set performance at each epoch -----


            for (int i = 0; i < nEpochs; i++) {
                net.fit(trainData);

                //Run regression evaluation on our single column input
                RegressionEvaluation evaluation = new RegressionEvaluation(numOutput);
                INDArray features = testData.getFeatureMatrix();

                INDArray lables = testData.getLabels();
                INDArray predicted = net.output(features, false);

                evaluation.evalTimeSeries(lables, predicted);

                //Just do sout here since the logger will shift the shift the columns of the stats
                System.out.println(evaluation.stats());
            }

            ModelSerializer.writeModel(net, locationToSave, saveUpdater);
        }else{
            net=ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        }


        //Init rrnTimeStemp with train data and predict test data
        net.rnnTimeStep(trainData.getFeatureMatrix());
        INDArray predicted = net.rnnTimeStep(testData.getFeatureMatrix());

        //Revert data back to original values for plotting
        normalizer.revert(trainData);
        normalizer.revert(testData);
        normalizer.revertLabels(predicted);

        //Create plot with out data
        XYSeriesCollection c = new XYSeriesCollection();
        createSeries(c, testData.getLabels(), 0, "Actual test data",numOutput==1);
        createSeries(c, predicted, 0, "Predicted test data",numOutput==1);

        plotDataset(c);

    }

    private static XYSeriesCollection createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name,boolean single) {
        int nRows = data.shape()[2];
        int repeat = data.shape()[1];

        for (int j = 0; j < repeat; j++) {
            XYSeries series = new XYSeries(name + j);
            for (int i = 0; i < nRows; i++) {
                if (!single)
                    series.add(i + offset, data.slice(0).slice(j).getDouble(i));
                else
                    series.add(i + offset, data.slice(j).getDouble(i));
            }
            seriesCollection.addSeries(series);
        }

        return seriesCollection;
    }

    /**
     * Generate an xy plot of the datasets provided.
     */
    private static void plotDataset(XYSeriesCollection c) {
        String xAxisLabel = "out";
        String yAxisLabel = "data";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        JFreeChart chart = ChartFactory.createXYLineChart("", xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

        // get a reference to the plot for further customisation...
        final XYPlot plot = chart.getXYPlot();

        // Auto zoom to fit time series in initial window
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);

        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        RefineryUtilities.centerFrameOnScreen(f);
        f.setVisible(true);
    }
}
