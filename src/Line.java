import java.awt.Color;


public class Line {
    public Color color;
    public double[] data;
    public boolean oneDim;
    public boolean event;
    
    
    public Line(Color col, double[] data){
    	this.oneDim=true;
    	this.color = col;
    	this.data=data;
    }
    public Line(Color col, double[] data,boolean oneDim){
    	this.oneDim=oneDim;
    	this.color = col;
    	this.data=data;
    }
    public Line(Color col, double[] data,boolean oneDim, boolean event){
    	this.oneDim=oneDim;
    	this.color = col;
    	this.data=data;
    	this.event=event;
    	if(event){ //make sure data is between 0 and 1
    		double min=0;
    		double max = 1;
    		for(int i=0; i<data.length;i++){
    			if(data[i]<min)min=data[i];
    			if(data[i]>max)max=data[i];
    		}
    		for(int i=0; i<data.length;i++){
    			data[i]-=min;
    			data[i]/=max-min;
    			if(data[i]<0||data[i]>1){System.out.println("Should never happen - Line Exit");System.exit(0);}
    		}
    		
    	}
    }
   
	

}
