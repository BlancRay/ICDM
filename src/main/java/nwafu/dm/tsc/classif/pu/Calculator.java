package nwafu.dm.tsc.classif.pu;

public class Calculator extends Thread{
	Thread thread = new Thread() { 
        @Override public void run() {
           System.out.println(">>> I am running in a separate thread!");
        }
   };
   
//   thread.start();
//   thread.join();
}