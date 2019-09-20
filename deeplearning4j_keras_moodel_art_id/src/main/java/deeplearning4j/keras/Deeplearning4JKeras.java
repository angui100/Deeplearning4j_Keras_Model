package deeplearning4j.keras;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.IOException;
import java.util.Arrays;


public class Deeplearning4JKeras {

    private MultiLayerNetwork model;
    //private String[] labels = {"Angry", "Fear OR Surprise", "Calm"};
    //private int imgHeight = 48;
    //private int imgWidth = 48;
    //private int imgChannel = 1;

    private void loadKerasModel(String modelH5Path)
    {
        try {
            if (model == null) {
                model = KerasModelImport.importKerasSequentialModelAndWeights(modelH5Path);
            }
        }
        catch(InvalidKerasConfigurationException configex)
        {
            configex.printStackTrace();
        }
        catch (UnsupportedKerasConfigurationException unsupex)
        {
            unsupex.printStackTrace();
        }
        catch(IOException ioex)
        {
            ioex.printStackTrace();
        }
    }

    private INDArray runPrediction(String imgFilePath)
    {
        INDArray output = null;
        try{
            int imgHeight = 48;
            int imgWidth = 48;
            int imgChannel = 1;
            NativeImageLoader loader = new NativeImageLoader(imgHeight, imgWidth, imgChannel);

            INDArray image = loader.asMatrix(imgFilePath);
            DataNormalization scalar = new ImagePreProcessingScaler(0, 1);
            scalar.transform(image);
            output = model.output(image);
        }
        catch(Exception ex)
        {
            ex.printStackTrace();
        }

        return output;
    }

    private void interpretPrediction(INDArray outResult)
    {
        String[] labels = {"Angry", "Fear OR Surprise", "Calm"};
        if( !outResult.isEmpty())
        {
            System.out.println(Arrays.toString(labels));
            System.out.println(outResult.toString());
        }
        else{
            System.out.println("Unknown object.......");
        }
    }


    public static void main(String[] args)
    {
        try
        {
            Deeplearning4JKeras dpk = new Deeplearning4JKeras();
            String modelFilePath = "/somewhere/somedir/model.h5";
            dpk.loadKerasModel(modelFilePath);

            String imgFilePath = "/somewhere/somedir/image.png(jpeg)";
            dpk.interpretPrediction(dpk.runPrediction(imgFilePath));

        }
        catch(Exception ex)
        {
            System.out.println(" ");
            System.out.println("---------------------------------------------");
            System.out.println(ex.toString());
            System.out.println("---------------------------------------------");
        }
    }
}
