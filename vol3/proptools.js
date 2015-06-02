var XOR_x = [
	[0,0],
	[1,0],
	[0,1],
	[1,1]
];

var XOR_y = [0,1,1,0];

function randomize() {
	w = [];
	for(var i=0;i<9;i++) {
		w.push((Math.random()*2)-1);
	}

	$("#w1").val(w[0]);
	$("#w2").val(w[1]);
	$("#w3").val(w[2]);
	$("#w4").val(w[3]);
	$("#w5").val(w[4]);
	$("#w6").val(w[5]);
	$("#w7").val(w[6]);
	$("#w8").val(w[7]);
	$("#w9").val(w[8]);

	$("#mse").html("mse: "+calculate_mse(w));
}

function calculate(i,w) {
	// Calculate hidden neuron 1
	var sum1 = (i[0]*w[0]) + (i[1]*w[2]) + w[4];
	var n1 = 1.0 / (1.0 + Math.exp(-sum1));
	// Calculate hidden neuron 2
	var sum2 = (i[0]*w[1]) + (i[1]*w[3]) + w[5];
	var n2 = 1.0 / (1.0 + Math.exp(-sum2));
	// Calculate output neuron
	var sum3 = (n1*w[6]) + (n2*w[7]) + w[8];
	return(1.0 / (1.0 + Math.exp(-sum3)));
}

function calculate_mse(w) {
	var d1 = calculate([0,0],w)-0.0;
	var d2 = calculate([1,0],w)-1.0;
	var d3 = calculate([0,1],w)-1.0;
	var d4 = calculate([1,1],w)-0.0;
	return ((d1*d1)+(d2*d2)+(d3*d3)+(d4*d4))/4;
}

function sigmoid(x) {
	return(1/(1+Math.exp(-x)));
}

function dSigmoid(x) {
	return(sigmoid(x)*(1-sigmoid(x)));
}

function round(d) {
	return(Math.round(d*100000)/100000);
}

function calcGradients(w,input,expected) {
	var output = "";
	output+="Training " + input + " to produce " + expected + "<br>";
	output+="<b>Feedforward: Calculate neural network output</b><br>";
	
	// Calculate hidden neuron 1
	var sum1 = (input[0]*w[0]) + (input[1]*w[2]) + w[4];
	output += "sum1=("+input[0]+"*"+w[0]+")+("+input[1]+"*"+w[2]+")+"+w[4]+"="+sum1+"<br>";

	var n1 = 1.0 / (1.0 + Math.exp(-sum1))
	output += "n1 = 1/(1+exp(-sum1)=" + n1+"<br>";

	$("#n1").val(n1)

	// Calculate hidden neuron 2
	var sum2 = (input[0]*w[1]) + (input[1]*w[3]) + w[5];
	output += "sum2=("+input[0]+"*"+w[1]+")+("+input[1]+"*"+w[3]+")+"+w[5]+"="+sum2+"<br>";

	var n2 = 1.0 / (1.0 + Math.exp(-sum2))
	output += "n2 = 1/(1+exp(-sum2)=" + n2+"<br>";

	$("#n2").val(n2)

	// Calculate output neuron
	var sum3 = (n1*w[6]) + (n2*w[7]) + w[8];
	output += "sum3=("+n1+"*"+w[6]+")+("+n2+"*"+w[7]+")+"+w[8]+"="+sum3+"<br>";

	var o = 1.0 / (1.0 + Math.exp(-sum3))
	output += "n3 = 1/(1+exp(-sum3)=" + o+"<br>";

	$("#o").val(o)

	output+="<b>Backpropagation</b><br>";
	output+="Note: sigmoid(x)=1/(1+exp(-x))<br>";
	output+="Note: dSigmoid(x)=sigmoid(x)*(1-sigmoid(x))<br>";
	o = calculate(input,w);
	output+="Currently, " + input + " produces output: " + o + "<br>";
	output+="Calculate: E = actual-ideal<br>";
	e = o - expected;
	output+="(" + o + "-" + expected + ")=" + e + "<br>";
	
	output+="<b>Backpropagation: Calculate Node (neuron) Deltas</b><br>";
	output+="<u>Calculate Node Delta For: O1</u><br>";
	output+="o1_nd = -e * dSigmoid(sum3)<br>";
	var o1nd = -e * dSigmoid(sum3);
	output+="o1_nd = " + (-e) + "*" + dSigmoid(sum3) + "=" + o1nd + "<br>";
	output+="<u>Calculate Node Delta For: H1</u><br>";
	var h1nd = dSigmoid(sum1)*o1nd*w[6];
	output+="h1_nd = dSigmoid(sum1) * (weight H1->O1) * o1nd<br>";
	output+="h1_nd = " + dSigmoid(sum1) + "*" + o1nd + "*" + w[6]+ "=" + h1nd+"<br>";
	output+="<u>Calculate Node Delta For: H2</u><br>";
	var h2nd = dSigmoid(sum2)*o1nd*w[7];
	output+="h2_nd = dSigmoid(sum2) * (weight H2->O1) * o1nd<br>";
	output+="h2_nd = " + dSigmoid(sum2) + "*" + o1nd + "*" + w[7]+ "=" + h2nd + "<br>";
	
	var g = Array.apply(null, new Array(9)).map(Number.prototype.valueOf,0);
	
	output+="<b>Backpropagation: Calculate Gradients</b><br>";
	g[0] = n1 * o1nd;
	output+="g(H1 to O1)=h1*o1_nd="+n1+"*"+o1nd+"= "+g[0]+"<br>";
	g[1] = n2 * o1nd;
	output+="g(H2 to O1)=h2*o1_nd="+n2+"*"+o1nd+"= "+g[1]+"<br>";
	g[2] = o1nd;
	output+="g(bias to O1)=1*o1_nd=1*"+o1nd+"= "+g[2]+"<br>";
	
	g[3] = input[0] * h1nd;
	output+="g(I1 to H1)=i1*h1_nd="+input[0]+"*"+h1nd+"= "+g[3]+"<br>";
	g[4] = input[1] * h1nd;
	output+="g(I2 to H1)=i2*h1_nd="+input[1]+"*"+h1nd+"= "+g[4]+"<br>";
	g[5] = h1nd;
	output+="g(bias to H1)=1*h1_nd=1*"+h1nd+"= "+g[5]+"<br>";
	
	g[6] = input[0] * h2nd;
	output+="g(I1 to H2)=i1*h2_nd="+input[0]+"*"+h2nd+"= "+g[6]+"<br>";
	g[7] = input[1] * h2nd;
	output+="g(I2 to H2)=i2*h2_nd="+input[1]+"*"+h2nd+"= "+g[7]+"<br>";
	g[8] = h2nd;
	output+="g(bias to H2)=1*h2_nd=1*"+h1nd+"= "+g[8]+"<br>";
	
	// tables
	table_html = "<table><tr><th>Node</th><th>Sum</th><th>Output</th><th>Node Delta</th><tr>";
	table_html+= "<tr><td>Output 1</td><td>"+round(sum3)+"</td><td>"+round(o)+"</td><td>"+round(o1nd)+"</td></tr>";
	table_html+= "<tr><td>Hidden 1</td><td>"+round(sum1)+"</td><td>"+round(n1)+"</td><td>"+round(h1nd)+"</td></tr>";
	table_html+= "<tr><td>Hidden 2</td><td>"+round(sum2)+"</td><td>"+round(n2)+"</td><td>"+round(h2nd)+"</td></tr>";
	table_html+= "</table>";
	
	return [g,table_html+output];
}

function trainOnline(input,expected) {
	var w = [];
	w.push(parseFloat($("#w1").val()));
	w.push(parseFloat($("#w2").val()));
	w.push(parseFloat($("#w3").val()));
	w.push(parseFloat($("#w4").val()));
	w.push(parseFloat($("#w5").val()));
	w.push(parseFloat($("#w6").val()));
	w.push(parseFloat($("#w7").val()));
	w.push(parseFloat($("#w8").val()));
	w.push(parseFloat($("#w9").val()));
	
	result = calcGradients(w,input,expected);
	g = result[0];
	output = result[1];
	
	var lr = parseFloat($("#lr").val());
	
	for(var i=0;i<g.length;i++) {
		w[i]+=(g[i]*0.7);
	}

	// Display MSE
	$("#mse").html("MSE: "+calculate_mse(w));

	// Display weights
	parseFloat($("#w1").val(w[0]));
	parseFloat($("#w2").val(w[1]));
	parseFloat($("#w3").val(w[2]));
	parseFloat($("#w4").val(w[3]));
	parseFloat($("#w5").val(w[4]));
	parseFloat($("#w6").val(w[5]));
	parseFloat($("#w7").val(w[6]));
	parseFloat($("#w8").val(w[7]));
	parseFloat($("#w9").val(w[8]));
	
	table_html= "<table><tr><th>Connection</th><th>Gradient</th><th>Old Weight</th><th>Delta</th><th>New Weight</th></tr>";
	table_html+= "<tr><td>0: H1->O1</td><td>"+round(g[0])+"</td><td>"+round(w[0])+"</td><td>"+round(g[0]*lr)+"</td><td>"+round(w[0]+(g[0]*0.7))+"</td></tr>";
	table_html+= "<tr><td>1: H2->O1</td><td>"+round(g[1])+"</td><td>"+round(w[1])+"</td><td>"+round(g[1]*lr)+"</td><td>"+round(w[1]+(g[1]*0.7))+"</td></tr>";
	table_html+= "<tr><td>2: B2->O1</td><td>"+round(g[2])+"</td><td>"+round(w[2])+"</td><td>"+round(g[2]*lr)+"</td><td>"+round(w[2]+(g[2]*0.7))+"</td></tr>";
	table_html+= "<tr><td>3: I1->H1</td><td>"+round(g[3])+"</td><td>"+round(w[3])+"</td><td>"+round(g[3]*lr)+"</td><td>"+round(w[3]+(g[3]*0.7))+"</td></tr>";
	table_html+= "<tr><td>4: I2->H1</td><td>"+round(g[4])+"</td><td>"+round(w[4])+"</td><td>"+round(g[4]*lr)+"</td><td>"+round(w[4]+(g[4]*0.7))+"</td></tr>";
	table_html+= "<tr><td>5: B1->H1</td><td>"+round(g[5])+"</td><td>"+round(w[5])+"</td><td>"+round(g[5]*lr)+"</td><td>"+round(w[5]+(g[5]*0.7))+"</td></tr>";
	table_html+= "<tr><td>6: I1->H2</td><td>"+round(g[6])+"</td><td>"+round(w[6])+"</td><td>"+round(g[6]*lr)+"</td><td>"+round(w[6]+(g[6]*0.7))+"</td></tr>";
	table_html+= "<tr><td>7: I2->H2</td><td>"+round(g[7])+"</td><td>"+round(w[7])+"</td><td>"+round(g[7]*lr)+"</td><td>"+round(w[7]+(g[7]*0.7))+"</td></tr>";
	table_html+= "<tr><td>8: B1->H2</td><td>"+round(g[8])+"</td><td>"+round(w[8])+"</td><td>"+round(g[8]*lr)+"</td><td>"+round(w[8]+(g[8]*0.7))+"</td></tr>";
	table_html+= "</table>";
		
	// Display calculation
	$("#calculationDisplay").html(table_html+output);
}

function trainBatch(prev_delta) {
	var w = [];
	w.push(parseFloat($("#w1").val()));
	w.push(parseFloat($("#w2").val()));
	w.push(parseFloat($("#w3").val()));
	w.push(parseFloat($("#w4").val()));
	w.push(parseFloat($("#w5").val()));
	w.push(parseFloat($("#w6").val()));
	w.push(parseFloat($("#w7").val()));
	w.push(parseFloat($("#w8").val()));
	w.push(parseFloat($("#w9").val()));
	
	var lr = parseFloat($("#lr").val());
	var mom = parseFloat($("#m").val());
	var output = "";
	
	batch_g = [0,0,0,0,0,0,0,0,0]
	
	for(var t=0;t<XOR_x.length;t++) {
		output+="<h3>Training Element #"+(t+1)+"</h3>";
		var x = XOR_x[t];
		var y = XOR_y[t];
	
		result = calcGradients(w,x,y);
		g = result[0];
		
		for(var i=0;i<g.length;i++) {
			batch_g[i]+=g[i]
		}
		
		output+= "<table><tr><th>Connection</th><th>Gradient</th><th>Sum of Gradients So Far</th></tr>";
		output+= "<tr><td>0: H1->O1</td><td>"+round(g[0])+"</td><td>"+round(batch_g[0])+"</td></tr>";
		output+= "<tr><td>1: H2->O1</td><td>"+round(g[1])+"</td><td>"+round(batch_g[1])+"</td></tr>";
		output+= "<tr><td>2: B2->O1</td><td>"+round(g[2])+"</td><td>"+round(batch_g[2])+"</td></tr>";
		output+= "<tr><td>3: I1->H1</td><td>"+round(g[3])+"</td><td>"+round(batch_g[3])+"</td></tr>";
		output+= "<tr><td>4: I2->H1</td><td>"+round(g[4])+"</td><td>"+round(batch_g[4])+"</td></tr>";
		output+= "<tr><td>5: B1->H1</td><td>"+round(g[5])+"</td><td>"+round(batch_g[5])+"</td></tr>";
		output+= "<tr><td>6: I1->H2</td><td>"+round(g[6])+"</td><td>"+round(batch_g[6])+"</td></tr>";
		output+= "<tr><td>7: I2->H2</td><td>"+round(g[7])+"</td><td>"+round(batch_g[7])+"</td></tr>";
		output+= "<tr><td>8: B1->H2</td><td>"+round(g[8])+"</td><td>"+round(batch_g[8])+"</td></tr>";
		output+= "</table>";
		
		output += result[1];
	}
	
	delta = [];
	for(var i=0;i<batch_g.length;i++) {
		delta.push( (batch_g[i]*lr)) // + (prev_delta[i]*mom) );
	}
	
	w2 = [];
	for(var i=0;i<batch_g.length;i++) {
		w2.push(delta[i]+w[i]);
	}
	
	output+="<h3>Weight Update</h3>";
	output+= "<table><tr><th>Connection</th><th>Gradient(batch)</th><th>Weight</th><th>Delta</th><th>Prev. Delta</th><th>New Weight</th></tr>";
	output+= "<tr><td>0: H1->O1</td><td>"+round(batch_g[0])+"</td><td>"+round(w[0])+"</td><td>"+round(delta[0])+"</td><td>"+round(prev_delta[0])+"</td><td>"+round(w2[0])+"</td></tr>";
	output+= "<tr><td>1: H2->O1</td><td>"+round(batch_g[1])+"</td><td>"+round(w[1])+"</td><td>"+round(delta[1])+"</td><td>"+round(prev_delta[1])+"</td><td>"+round(w2[1])+"</td></tr>";
	output+= "<tr><td>2: B2->O1</td><td>"+round(batch_g[2])+"</td><td>"+round(w[2])+"</td><td>"+round(delta[2])+"</td><td>"+round(prev_delta[2])+"</td><td>"+round(w2[2])+"</td></tr>";
	output+= "<tr><td>3: I1->H1</td><td>"+round(batch_g[3])+"</td><td>"+round(w[3])+"</td><td>"+round(delta[3])+"</td><td>"+round(prev_delta[3])+"</td><td>"+round(w2[3])+"</td></tr>";
	output+= "<tr><td>4: I2->H1</td><td>"+round(batch_g[4])+"</td><td>"+round(w[4])+"</td><td>"+round(delta[4])+"</td><td>"+round(prev_delta[4])+"</td><td>"+round(w2[4])+"</td></tr>";
	output+= "<tr><td>5: B1->H1</td><td>"+round(batch_g[5])+"</td><td>"+round(w[5])+"</td><td>"+round(delta[5])+"</td><td>"+round(prev_delta[5])+"</td><td>"+round(w2[5])+"</td></tr>";
	output+= "<tr><td>6: I1->H2</td><td>"+round(batch_g[6])+"</td><td>"+round(w[6])+"</td><td>"+round(delta[6])+"</td><td>"+round(prev_delta[6])+"</td><td>"+round(w2[6])+"</td></tr>";
	output+= "<tr><td>7: I2->H2</td><td>"+round(batch_g[7])+"</td><td>"+round(w[7])+"</td><td>"+round(delta[7])+"</td><td>"+round(prev_delta[7])+"</td><td>"+round(w2[7])+"</td></tr>";
	output+= "<tr><td>8: B1->H2</td><td>"+round(batch_g[8])+"</td><td>"+round(w[8])+"</td><td>"+round(delta[8])+"</td><td>"+round(prev_delta[8])+"</td><td>"+round(w2[8])+"</td></tr>";
	output+= "</table>";
	
	for(var i=0;i<g.length;i++) {
		w[i]+=delta[i];
		prev_delta[i]=delta[i];
	}

	// Display MSE
	$("#mse").html("MSE: "+calculate_mse(w));

	// Display weights
	parseFloat($("#w1").val(w[0]));
	parseFloat($("#w2").val(w[1]));
	parseFloat($("#w3").val(w[2]));
	parseFloat($("#w4").val(w[3]));
	parseFloat($("#w5").val(w[4]));
	parseFloat($("#w6").val(w[5]));
	parseFloat($("#w7").val(w[6]));
	parseFloat($("#w8").val(w[7]));
	parseFloat($("#w9").val(w[8]));
		
	// Display calculation
	$("#calculationDisplay").html(output);
}