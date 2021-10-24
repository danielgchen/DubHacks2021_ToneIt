package com.company;

import java.io.File;
        import java.io.FileNotFoundException;
        import java.io.PrintStream;
        import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws FileNotFoundException {
        File file = new File("train.txt");
        Scanner input = new Scanner(file);
//        int i = 1;
        PrintStream output = new PrintStream(new File("trainTone.txt"));
        output.println("[");
        while (input.hasNextLine()) {
            String line = input.nextLine();
            Scanner lineInput = new Scanner(line).useDelimiter(";");

            String sentence = lineInput.next();
            String tone = lineInput.next();

            output.print("{" + " \"sentence\": " + "\"" + sentence + "\"" + ", \"tone\": " + "\"" + tone + "\"" + "},");
            output.println("");
        }
        output.println("]");
        output.close();

    }
}
