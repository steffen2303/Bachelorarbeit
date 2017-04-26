import java.awt.Color;


public class Event {
	public String name;
	public int varnum;
	public int value;
	public boolean uperbound;
	public Color color;
	
	public Event(int n,int v){
		this.name = v + " überschritten";
		this.varnum = n;
		this.value = v;
		this.uperbound = true;
		this.color = Color.gray;
	}
	public Event(String s,int n,int v, boolean h){
		this.name = s;
		this.varnum = n;
		this.value = v;
		this.uperbound = h;
		this.color= Color.gray;
	}
	public Event(String s,int n,int v, boolean h, Color c){
		this.name = s;
		this.varnum = n;
		this.value = v;
		this.uperbound = h;
		this.color= c;
	}
}
