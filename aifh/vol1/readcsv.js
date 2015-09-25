$(document).ready(function(){
  $("#btnParse").click(function(){
    contents = $.csv.toArrays($("#textCSV").val());
    var tableStr = "<table><thead></tr>";
    for(var col=0;col<contents[0].length;col++) {
		tableStr+="<th>";
    	tableStr+=contents[0][col];
    	tableStr+="</th>";
    }
    tableStr+="</tr></thead><tbody>";
    for(var row=1; row<contents.length; row++) {
    	tableStr+="<tr>";
    	for(var col=0;col<contents[row].length;col++) {
    		tableStr+="<td>";
    		tableStr+=contents[row][col];
    		tableStr+="</td>";
    	}
    	tableStr+="</tr>";
    }
    tableStr+="</tbody></table>";
    $("#example-output").html(tableStr);
  });
  
  $("#btnPreload").click(function(){
  	var url = $("#example-preload").val();
  	$.get(url, 
    	function(data) {
        	$("#textCSV").val(data);
        });
  	
  })
});

