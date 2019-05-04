package RgStEb.Base;

import java.util.Arrays;
import java.util.HashSet;

import weka.core.*;

public class ConvertToWeka {

    private Instances wekaInstances;

    public ConvertToWeka() {
        System.out.println("Constructor!");
    }

    public static String test() {
        return "hello";
    }


//    classLabel: all the classes of all instances
//    public ConvertToWeka(String name, String attrNameNumeric[], double dataNumeric[][], String classLabel[], boolean hasClass) {
//        int numAttrNumeric = attrNameNumeric.length;    // number of features
//        int numAttr = numAttrNumeric;                    // numAttr: number of all attributes (features + class)
//        if (hasClass)
//            numAttr++; // with the class label
//        int numDataNumeric = dataNumeric[0].length;
//        int numData = numDataNumeric;                    // number of instances
//
//        FastVector vec = new FastVector(numAttrNumeric + 1);
//        // adding data attributes
//        for (int i = 0; i < numAttrNumeric; i++)
//            vec.addElement(new Attribute(attrNameNumeric[i]));
//
//        // adding the class label
//        if (hasClass) {
//            // getting the unique strings within the classLabel.
//            String[] uClasslabel = new HashSet<String>(Arrays.asList(classLabel)).toArray(new String[0]);
//            FastVector classValues = new FastVector(uClasslabel.length);
//
//            for (int j = 0; j < uClasslabel.length; j++)
//                classValues.addElement(uClasslabel[j]);
//            vec.addElement(new Attribute("class", classValues));
//        }
//
//        wekaInstances = new Instances(name, vec, numData);
//        // adding data values
//        for (int i = 0; i < numData; i++) {
//            Instance inst = new DenseInstance(numAttr);
//            // adding numeric value
//            for (int j = 0; j < numAttrNumeric; j++) {
//                inst.setDataset(wekaInstances);
//                inst.setValue(j, dataNumeric[j][i]);
//            }
//
//            // adding class value
//            if (hasClass) {
//                inst.setDataset(wekaInstances);
//                inst.setValue(numAttr - 1, classLabel[i]);
//            }
//            wekaInstances.add(inst);
//        }
//        if (hasClass)
//            wekaInstances.setClassIndex(numAttr - 1);
//    }

    public ConvertToWeka(String name, String attrNameNumeric[], double dataNumeric[][], String classLabel[], boolean hasClass) {
        int numAttrNumeric = attrNameNumeric.length;
        int numAttr = numAttrNumeric;
        if (hasClass)
            numAttr++; // with the class label
        int numDataNumeric = dataNumeric[0].length;
        int numData = numDataNumeric;

        FastVector vec = new FastVector(numAttrNumeric + 1);
        // adding data attributes
        for (int i = 0; i < numAttrNumeric; i++)
            vec.addElement(new Attribute(attrNameNumeric[i]));

        // adding the class label
        if (hasClass) {
            // getting the unique strings within the classLabel.
            String[] uClasslabel = new HashSet<String>(Arrays.asList(classLabel)).toArray(new String[0]);
            FastVector classValues = new FastVector(uClasslabel.length);

            for (int j = 0; j < uClasslabel.length; j++)
                classValues.addElement(uClasslabel[j]);
            vec.addElement(new Attribute("class", classValues));
        }

        wekaInstances = new Instances(name, vec, numData);
        // adding data values
        for (int i = 0; i < numData; i++) {
            Instance inst = new Instance(numAttr);
            // adding numeric value
            for (int j = 0; j < numAttrNumeric; j++) {
                inst.setDataset(wekaInstances);
                inst.setValue(j, dataNumeric[j][i]);
            }

            // adding class value
            if (hasClass) {
                inst.setDataset(wekaInstances);
                inst.setValue(numAttr - 1, classLabel[i]);
            }
            wekaInstances.add(inst);
        }
        if (hasClass)
            wekaInstances.setClassIndex(numAttr - 1);
    }


    public Instances getInstances() {
        return wekaInstances;
    }

}

