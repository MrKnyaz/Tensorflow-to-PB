import org.tensorflow.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class LoadTensorflowModel {

    public static void main(String[] args) throws Exception {
        System.out.println("TensorFlow version: " + TensorFlow.version());
        //Get absolute path to src/main/resources/saved_model.pb
        Path modelPath = Paths.get(LoadTensorflowModel.class.getResource("saved_model.pb").toURI());
        byte[] graph = Files.readAllBytes(modelPath);

        try (Graph g = new Graph()) {
            g.importGraphDef(graph);

            //Just print needed operations for debug
            System.out.println(g.operation("input").output(0));
            System.out.println(g.operation("not_activated_output").output(0));

            //open session using imported graph
            try (Session sess = new Session(g)) {
                float[][] inputData = {{4, 3, 2, 1}};

                // We have to create tensor to feed it to session,
                // unlike in Python where you just pass Numpy array
                Tensor inputTensor = Tensor.create(inputData, Float.class);
                float[][] output = predict(sess, inputTensor);
                for (int i = 0; i < output[0].length; i++) {
                    System.out.println(output[0][i]);
                }
            }
        }
    }

    private static float[][] predict(Session sess, Tensor inputTensor) {
        Tensor result = sess.runner()
                .feed("input", inputTensor)
                .fetch("not_activated_output").run().get(0);
        float[][] outputBuffer = new float[1][3];
        result.copyTo(outputBuffer);
        return outputBuffer;
    }

}
