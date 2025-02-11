import { useSelector } from 'react-redux';

import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline, StyledEngineProvider } from '@mui/material';

// routing
import Routes from 'routes';

// defaultTheme
import themes from 'themes';

// project imports
import NavigationScroll from 'layout/NavigationScroll';
import {ParamProvider} from 'context';

import { ChakraProvider } from '@chakra-ui/react'



// ==============================|| APP ||============================== //

const App = () => {
  const customization = useSelector((state) => state.customization);

  return (
    <StyledEngineProvider injectFirst>
      <ChakraProvider>
      <ThemeProvider theme={themes(customization)}>
        <ParamProvider>
          <CssBaseline />
          <NavigationScroll>
            
              <Routes />
          </NavigationScroll>
        </ParamProvider>
      </ThemeProvider>
      </ChakraProvider>
    </StyledEngineProvider>
  );
};

export default App;
