package wekajava;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import java.awt.*;
import java.text.DecimalFormat;
import java.util.Random;
import weka.classifiers.AbstractClassifier;

public class Wekajava {

    // ---------- MÉTODO PRINCIPAL ----------
    public static void main(String[] args) {
        try {
            /* === 1. Cargar datos === */
            String arffPath = "C:\\Users\\PIERO\\Downloads\\calificaciones_unidas_nominalizadas.arff";
            DataSource source = new DataSource(arffPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);  // la clase es la última columna

            /* === 2. Definir clasificadores === */
            Classifier ds = new DecisionStump();
            Classifier nb = new NaiveBayes();

            /* === 3. Evaluación con 10-fold cross-validation (métrica global) === */
            Evaluation evalDS = new Evaluation(data);
            evalDS.crossValidateModel(ds, data, 10, new Random(1));

            Evaluation evalNB = new Evaluation(data);
            evalNB.crossValidateModel(nb, data, 10, new Random(1));

            /* === 4. Mostrar las matrices de confusión (visual) === */
            String[] clases = new String[data.numClasses()];
            for (int i = 0; i < data.numClasses(); i++) {
                clases[i] = data.classAttribute().value(i);
            }
            mostrarMatrizConfusion(evalDS.confusionMatrix(), clases,
                                   "Matriz de Confusión – Decision Stump");
            mostrarMatrizConfusion(evalNB.confusionMatrix(), clases,
                                   "Matriz de Confusión – Naive Bayes");

            /* === 5. Validación cruzada manual para obtener accuracy por fold === */
            int folds = 10;
            double[] accDS = obtenerAccuraciesPorFold(data, new DecisionStump(), folds);
            double[] accNB = obtenerAccuraciesPorFold(data, new NaiveBayes(),   folds);

            /* === 6. Mostrar accuracy por fold (visual) === */
            mostrarAccuracyPorFold(accDS, "Accuracy por Fold – Decision Stump");
            mostrarAccuracyPorFold(accNB, "Accuracy por Fold – Naive Bayes");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ---------- OBTENER ACCURACY POR FOLD ----------
    private static double[] obtenerAccuraciesPorFold(Instances data,
                                                     Classifier baseCls,
                                                     int folds) throws Exception {
        Instances randData = new Instances(data);
        randData.randomize(new Random(1));
        if (randData.classAttribute().isNominal()) {
            randData.stratify(folds);
        }

        double[] acc = new double[folds];
        DecimalFormat df = new DecimalFormat("#.##");

        for (int i = 0; i < folds; i++) {
            Instances train = randData.trainCV(folds, i);
            Instances test  = randData.testCV(folds,  i);

            Classifier cls = AbstractClassifier.makeCopy(baseCls);
            cls.buildClassifier(train);

            Evaluation ev = new Evaluation(train);
            ev.evaluateModel(cls, test);
            acc[i] = Double.parseDouble(df.format(ev.pctCorrect()));
        }
        return acc;
    }

    // ---------- MOSTRAR MATRIZ DE CONFUSIÓN EN UN JTable ----------
    private static void mostrarMatrizConfusion(double[][] matrix,
                                               String[] clases,
                                               String title) {
        int n = clases.length;

        // Encabezados de columnas
        String[] colHeaders = new String[n + 1];
        colHeaders[0] = "Real \\ Pred.";
        System.arraycopy(clases, 0, colHeaders, 1, n);

        // Datos de la tabla
        String[][] datos = new String[n][n + 1];
        DecimalFormat df = new DecimalFormat("#");

        for (int i = 0; i < n; i++) {
            datos[i][0] = clases[i];
            for (int j = 0; j < n; j++) {
                datos[i][j + 1] = df.format(matrix[i][j]);
            }
        }

        JTable table = new JTable(datos, colHeaders);
        table.setEnabled(false);
        table.setRowHeight(28);

        // Colorear diagonal (aciertos) y off-diagonal (errores)
        table.setDefaultRenderer(Object.class, new DefaultTableCellRenderer() {
            @Override
            public Component getTableCellRendererComponent(JTable tbl, Object value,
                                                           boolean isSel, boolean hasFocus,
                                                           int row, int col) {
                Component c = super.getTableCellRendererComponent(tbl, value,
                                                                  isSel, hasFocus, row, col);
                if (col > 0) {  // Ignorar columna de etiquetas
                    if (row == col - 1) {
                        c.setBackground(new Color(198, 239, 206)); // verde claro
                    } else {
                        c.setBackground(new Color(255, 199, 206)); // rojo claro
                    }
                } else {
                    c.setBackground(Color.WHITE);                 // etiqueta fila
                }
                setHorizontalAlignment(CENTER);
                return c;
            }
        });

        JScrollPane sp = new JScrollPane(table);
        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.add(sp);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    // ---------- MOSTRAR ACCURACY POR FOLD EN UN JTable ----------
    private static void mostrarAccuracyPorFold(double[] acc, String title) {
        String[] colHeaders = {"Fold", "Accuracy (%)"};
        String[][] datos = new String[acc.length][2];

        for (int i = 0; i < acc.length; i++) {
            datos[i][0] = "Fold " + (i + 1);
            datos[i][1] = String.format("%.2f", acc[i]);
        }

        JTable table = new JTable(datos, colHeaders);
        table.setEnabled(false);
        table.setRowHeight(26);

        JScrollPane sp = new JScrollPane(table);
        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.add(sp);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}