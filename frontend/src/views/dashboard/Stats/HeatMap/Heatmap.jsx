/* eslint-disable */

import * as React from 'react';
import HeatMap from "../../../../ui-component/Heatmap/index";
// import List from '@mui/material/List';
// import ListItem from '@mui/material/ListItem';
// import ListItemText from '@mui/material/ListItemText';
// import ListSubheader from '@mui/material/ListSubheader';
// import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';


import { ParametersContext } from 'context';

// Display only even labels
const xLabelsVisibility = new Array(24)
  .fill(0)
  .map((_, i) => (i % 1 === 0 ? true : false));

export function Heatmap() {

  const { a, outputList, X, setX,Y, setY, CellClick, data } = React.useContext(ParametersContext);
  
  return (
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <div style={{ fontSize: "13px", marginTop:"100px" }}>
            <HeatMap
              xLabels={data.xlabels}
              yLabels={data.ylabels}
              xLabelWidth={6}
              yLabelWidth={80}
              xLabelsLocation={"top"}
              xLabelsVisibility={xLabelsVisibility}
              data={data.data}
              squares
              height={45}
              onClick={CellClick}
                cellStyle={(background, value, min, max) => ({
                background: `rgb(0, 151, 230, ${1 - (max - value) / (max - min)})`,
                fontSize: "11.5px",
                color: "#444"
              })}
              cellRender={value => value && <div>{value}</div>}
            />
          </div>
        </Grid>
        
    </Grid>
  );
}
