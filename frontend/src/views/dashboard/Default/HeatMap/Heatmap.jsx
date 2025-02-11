/* eslint-disable */
import * as React from 'react';
import HeatMap from "../../../../ui-component/Heatmap/index";
import Grid from '@mui/material/Grid';
import { ParametersContext } from 'context';

const xLabelsVisibility = new Array(24).fill(0).map((_, i) => (i % 1 === 0 ? true : false));

export function Heatmap() {
  const { url } = React.useContext(ParametersContext);
  
  return (
      <Grid container spacing={2}>
        <Grid item xs={12}> 
          <div style={{ fontSize: "13px", marginTop:"100px" }}>
            hererererrerere
          </div>
        </Grid>
      </Grid>
  );
}
