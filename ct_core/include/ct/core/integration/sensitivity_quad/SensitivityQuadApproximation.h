/**********************************************************************************************************************
This file is part of the Control Toolbox (https://github.com/ethz-adrl/control-toolbox), copyright by ETH Zurich.
Licensed under the BSD-2 license (see LICENSE file in main directory)
**********************************************************************************************************************/

#pragma once

#include <unsupported/Eigen/MatrixFunctions>

#define SYMPLECTIC_ENABLED        \
    template <size_t V, size_t P> \
    typename std::enable_if<(V > 0 && P > 0), void>::type
#define SYMPLECTIC_DISABLED       \
    template <size_t V, size_t P> \
    typename std::enable_if<(V <= 0 || P <= 0), void>::type

namespace ct {
namespace core {

//! interface class for a general quadratic system or quadratized system
/*!
 * Defines the interface for a quadratic system
 *
 * \tparam STATE_DIM size of state vector
 * \tparam CONTROL_DIM size of input vector
 */
template <size_t STATE_DIM,
    size_t CONTROL_DIM,
    size_t P_DIM = STATE_DIM / 2,
    size_t V_DIM = STATE_DIM / 2,
    typename SCALAR = double>
class SensitivityQuadApproximation : public SensitivityQuad<STATE_DIM, CONTROL_DIM, SCALAR>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef StateMatrix<STATE_DIM, SCALAR> state_matrix_t;                              //!< state Jacobian type
    typedef StateControlMatrix<STATE_DIM, CONTROL_DIM, SCALAR> state_control_matrix_t;  //!< input Jacobian type


    //! constructor
    SensitivityQuadApproximation(const SCALAR& dt,
        const std::shared_ptr<QuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>>& quadraticSystem = nullptr,
        const SensitivityQuadApproximationSettings::APPROXIMATION& approx =
            SensitivityQuadApproximationSettings::APPROXIMATION::FORWARD_EULER)
        : quadraticSystem_(quadraticSystem), settings_(dt, approx)
    {
    }


    //! constructor
    SensitivityQuadApproximation(const SensitivityQuadApproximationSettings& settings,
        const std::shared_ptr<QuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>>& quadraticSystem = nullptr)
        : quadraticSystem_(quadraticSystem), settings_(settings)
    {
    }


    //! copy constructor
    SensitivityQuadApproximation(const SensitivityQuadApproximation& other) : settings_(other.settings_)
    {
        if (other.quadraticSystem_ != nullptr)
            quadraticSystem_ = std::shared_ptr<QuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>>(other.quadraticSystem_->clone());
    }


    //! destructor
    virtual ~SensitivityQuadApproximation(){};


    //! deep cloning
    virtual SensitivityQuadApproximation<STATE_DIM, CONTROL_DIM, P_DIM, V_DIM, SCALAR>* clone() const override
    {
        return new SensitivityQuadApproximation<STATE_DIM, CONTROL_DIM, P_DIM, V_DIM, SCALAR>(*this);
    }


    //! update the approximation type for the discrete-time system
    virtual void setApproximation(const SensitivityQuadApproximationSettings::APPROXIMATION& approx) override
    {
        settings_.approximation_ = approx;
    }


    //! retrieve the approximation type for the discrete-time system
    SensitivityQuadApproximationSettings::APPROXIMATION getApproximation() const { return settings_.approximation_; }
    //! update the quadratic system
    virtual void setQuadraticSystem(
        const std::shared_ptr<QuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>>& quadraticSystem) override
    {
        quadraticSystem_ = quadraticSystem;
    }


    //! update the time discretization
    virtual void setTimeDiscretization(const SCALAR& dt) override { settings_.dt_ = dt; }
    //! update the settings
    void updateSettings(const SensitivityQuadApproximationSettings& settings) { settings_ = settings; }

    //! get fxx, fxu, fuu, fx, and fu matrix for quadratic time invariant system
    /*!
     * @param x	the state setpoint
     * @param u the control setpoint
     * @param n the time setpoint
     * @param numSteps number of timesteps of trajectory for which to get the sensitivity for
     * @param fxx the resulting quadratic system matrix fxx
     * @param fxu the resulting quadratic system matrix fxu
     * @param fuu the resulting quadratic system matrix fuu
     * @param fx the resulting quadratic system matrix fx
     * @param fu the resulting quadratic system matrix fu
     */
    virtual void getQuadratizedDynamics(const StateVector<STATE_DIM, SCALAR>& x,
        const ControlVector<CONTROL_DIM, SCALAR>& u,
        const StateVector<STATE_DIM, SCALAR>& x_next,
        const int n,
        size_t numSteps,
        state_matrix_t& fxx,
        state_matrix_t& fxu,
        state_control_matrix_t& fuu,
        state_matrix_t& fx,
        state_control_matrix_t& fu) override
    {
        if (quadraticSystem_ == nullptr)
            throw std::runtime_error("Error in SensitivityQuadApproximation: quadraticSystem not properly set.");

        /*!
		 * fQr an QTI system fxx, fxu, fuu, fx, and fu won't change with time n, hence the quadratizations result from the following QTV special case.
         * TODO just use use the fx and fu for now
		 */
        switch (settings_.approximation_)
        {
            case SensitivityQuadApproximationSettings::APPROXIMATION::FORWARD_EULER:
            {
                forwardEuler(x, u, n, fx, fu);
                break;
            }
            case SensitivityQuadApproximationSettings::APPROXIMATION::BACKWARD_EULER:
            {
                backwardEuler(x, u, n + 1, fx, fu);
                break;
            }
            case SensitivityQuadApproximationSettings::APPROXIMATION::SYMPLECTIC_EULER:
            {
                symplecticEuler<V_DIM, P_DIM>(x, u, x_next, n, fx, fu);
                break;
            }
            case SensitivityQuadApproximationSettings::APPROXIMATION::TUSTIN:
            {
                /*!
                 * the Tustin (also known as 'Heun') approximation uses the state and control at the *start* and at the *end*
                 * of the ZOH interval to generate linear approximations A and B in a trapezoidal fashion.
                 */

                //! continuous-time A and B matrices
                state_matrix_t Ac_front;
                state_control_matrix_t Bc_front;

                // front derivatives
                quadraticSystem_->getDerivatives(Ac_front, Bc_front, x, u, n * settings_.dt_);
                Ac_front *= settings_.dt_;

                state_matrix_t Ac_back =
                    settings_.dt_ * quadraticSystem_->getDerivativeState(x_next, u, (n + 1) * settings_.dt_);


                //! tustin approximation
                state_matrix_t aNewInv;
                aNewInv.template topLeftCorner<STATE_DIM, STATE_DIM>() =
                    (state_matrix_t::Identity() - Ac_back).colPivHouseholderQr().inverse();
                fx = aNewInv * (state_matrix_t::Identity() + Ac_front);
                fu = aNewInv * settings_.dt_ * Bc_front;
                break;
            }
            case SensitivityQuadApproximationSettings::APPROXIMATION::MATRIX_EXPONENTIAL:
            {
                matrixExponential(x, u, n, fx, fu);
                break;
            }
            default:
                throw std::runtime_error("Unknown Approximation type in SensitivityQuadApproximation.");
        }  // end switch
    }

private:
    void forwardEuler(const StateVector<STATE_DIM, SCALAR>& x_n,
        const ControlVector<CONTROL_DIM, SCALAR>& u_n,
        const int& n,
        state_matrix_t& A_discr,
        state_control_matrix_t& B_discr)
    {
        /*!
		 * the Forward Euler approximation uses the state and control at the *start* of the ZOH interval to
		 * generate linear approximations A and B.
		 */
        state_matrix_t A_cont;
        state_control_matrix_t B_cont;
        quadraticSystem_->getDerivatives(A_cont, B_cont, x_n, u_n, n * settings_.dt_);

        A_discr = state_matrix_t::Identity() + settings_.dt_ * A_cont;
        B_discr = settings_.dt_ * B_cont;
    }

    void backwardEuler(const StateVector<STATE_DIM, SCALAR>& x_n,
        const ControlVector<CONTROL_DIM, SCALAR>& u_n,
        const int& n,
        state_matrix_t& A_discr,
        state_control_matrix_t& B_discr)
    {
        /*!
		 * the Backward Euler approximation uses the state and control at the *end* of the ZOH interval to
		 * generate linear approximations A and B.
		 */
        state_matrix_t A_cont;
        state_control_matrix_t B_cont;
        quadraticSystem_->getDerivatives(A_cont, B_cont, x_n, u_n, n * settings_.dt_);

        state_matrix_t aNew = settings_.dt_ * A_cont;
        A_discr.setZero();
        A_discr.template topLeftCorner<STATE_DIM, STATE_DIM>() =
            (state_matrix_t::Identity() - aNew).colPivHouseholderQr().inverse();

        B_discr = A_discr * settings_.dt_ * B_cont;
    }


    void matrixExponential(const StateVector<STATE_DIM, SCALAR>& x_n,
        const ControlVector<CONTROL_DIM, SCALAR>& u_n,
        const int& n,
        state_matrix_t& A_discr,
        state_control_matrix_t& B_discr)
    {
        state_matrix_t Ac;
        state_control_matrix_t Bc;
        quadraticSystem_->getDerivatives(Ac, Bc, x_n, u_n, n * settings_.dt_);

        state_matrix_t Adt = settings_.dt_ * Ac;

        A_discr.template topLeftCorner<STATE_DIM, STATE_DIM>() = Adt.exp();
        B_discr.template topLeftCorner<STATE_DIM, CONTROL_DIM>() =
            Ac.inverse() * (A_discr - state_matrix_t::Identity()) * Bc;
    }


    //!get the discretized quadratic system x^T fxx x + x^T fxu u + fx x + fu u corresponding for a symplectic integrator with full parameterization
    /*!
	 * @param x	state at start of interval
	 * @param u control at start of interval
	 * @param x_next state at end of interval
	 * @param u_next control at end of interval
	 * @param n time index
	 * @param fxx_sym resulting symplectic discrete-time fxx matrix
	 * @param fxu_sym resulting symplectic discrete-time fxu matrix
	 * @param fuu_sym resulting symplectic discrete-time fuu matrix
	 * @param fx_sym resulting symplectic discrete-time fx matrix
	 * @param fu_sym resulting symplectic discrete-time fu matrix
	 */
    SYMPLECTIC_ENABLED symplecticEuler(const StateVector<STATE_DIM, SCALAR>& x,
        const ControlVector<CONTROL_DIM, SCALAR>& u,
        const StateVector<STATE_DIM, SCALAR>& x_next,
        const int& n,
        state_matrix_t& fxx_sym,
        state_matrix_t& fxu_sym,
        state_control_matrix_t& fuu_sym,
        state_matrix_t& fx_sym,
        state_control_matrix_t& fu_sym)
    {
        const SCALAR& dt = settings_.dt_;

        // our implementation of symplectic integrators first updates the positions, we need to reconstruct an intermediate state accordingly
        StateVector<STATE_DIM, SCALAR> x_interm = x;
        x_interm.topRows(P_DIM) = x_next.topRows(P_DIM);

        state_matrix_t Ac1;          // continuous time A matrix for start state and control
        state_control_matrix_t Bc1;  // continuous time B matrix for start state and control
        quadraticSystem_->getDerivatives(Ac1, Bc1, x, u, n * dt);

        state_matrix_t Ac2;          // continuous time A matrix for intermediate state and control
        state_control_matrix_t Bc2;  // continuous time B matrix for intermediate state and control
        quadraticSystem_->getDerivatives(Ac2, Bc2, x_interm, u, n * dt);

        getSymplecticEulerApproximation<V_DIM, P_DIM>(Ac1, Ac2, Bc1, Bc2, fx_sym, fu_sym);
    }


    //!get the discretized quadraitc system x^T fxx x + x^T fxu u + fx x + fu u corresponding for a symplectic integrator with reduced
    /*!
	 * version without intermediate state. In this method, we do not consider the updated intermediate state from the symplectic integration step
	 * and compute the discretizd fxx, fxu, fuu, fx, and fu matrix using the continuous-time quadratization at the starting state and control only.
	 * \note this approximation is less costly but not 100% correct
	 * @param x	state at start of interval
	 * @param u control at start of interval
	 * @param n time index
	 * @param fxx_sym resulting symplectic discrete-time fxx matrix
	 * @param fxu_sym resulting symplectic discrete-time fxu matrix
	 * @param fuu_sym resulting symplectic discrete-time fuu matrix
	 * @param fx_sym resulting symplectic discrete-time fx matrix
	 * @param fu_sym resulting symplectic discrete-time fu matrix
     * 
	 */
    SYMPLECTIC_ENABLED symplecticEuler(const StateVector<STATE_DIM, SCALAR>& x,
        const ControlVector<CONTROL_DIM, SCALAR>& u,
        const int& n,
        state_matrix_t& fxx_sym,
        state_matrix_t& fxu_sym,
        state_control_matrix_t& fuu_sym,
        state_matrix_t& fx_sym,
        state_control_matrix_t& fu_sym)
    {
        const SCALAR& dt = settings_.dt_;

        state_matrix_t Ac1;          // continuous time A matrix for start state and control
        state_control_matrix_t Bc1;  // continuous time B matrix for start state and control
        quadraticSystem_->getDerivatives(Ac1, Bc1, x, u, n * dt);

        getSymplecticEulerApproximation<V_DIM, P_DIM>(Ac1, Ac1, Bc1, Bc1, fx_sym, fu_sym);
    }


    //! performs the symplectic Euler approximation
    /*!
	 * @param Ac1	continuous-time A matrix at start state and control
	 * @param Ac2	continuous-time A matrix at intermediate-step state and start control
	 * @param Bc1	continuous-time B matrix at start state and control
	 * @param Bc2	continuous-time B matrix at intermediate-step state and start control
	 * @param A_sym	resulting discrete-time symplectic A matrix
	 * @param B_sym resulting discrete-time symplectic B matrix
	 */
    SYMPLECTIC_ENABLED getSymplecticEulerApproximation(const state_matrix_t& Ac1,
        const state_matrix_t& Ac2,
        const state_control_matrix_t& Bc1,
        const state_control_matrix_t& Bc2,
        state_matrix_t& A_sym,
        state_control_matrix_t& B_sym)
    {
        const SCALAR& dt = settings_.dt_;

        typedef Eigen::Matrix<SCALAR, P_DIM, P_DIM> p_matrix_t;
        typedef Eigen::Matrix<SCALAR, V_DIM, V_DIM> v_matrix_t;
        typedef Eigen::Matrix<SCALAR, P_DIM, V_DIM> p_v_matrix_t;
        typedef Eigen::Matrix<SCALAR, V_DIM, P_DIM> v_p_matrix_t;
        typedef Eigen::Matrix<SCALAR, P_DIM, CONTROL_DIM> p_control_matrix_t;
        typedef Eigen::Matrix<SCALAR, V_DIM, CONTROL_DIM> v_control_matrix_t;

        // for ease of notation, make a block-wise map to references
        // elements taken form the linearization at the starting state
        const Eigen::Ref<const p_matrix_t> A11 = Ac1.topLeftCorner(P_DIM, P_DIM);
        const Eigen::Ref<const p_v_matrix_t> A12 = Ac1.topRightCorner(P_DIM, V_DIM);
        const Eigen::Ref<const p_control_matrix_t> B1 = Bc1.topRows(P_DIM);

        // elements taken from the linearization at the intermediate state
        const Eigen::Ref<const v_p_matrix_t> A21 = Ac2.bottomLeftCorner(V_DIM, P_DIM);
        const Eigen::Ref<const v_matrix_t> A22 = Ac2.bottomRightCorner(V_DIM, V_DIM);
        const Eigen::Ref<const v_control_matrix_t> B2 = Bc2.bottomRows(V_DIM);

        // discrete approximation A matrix
        A_sym.topLeftCorner(P_DIM, P_DIM) = p_matrix_t::Identity() + dt * A11;
        A_sym.topRightCorner(P_DIM, V_DIM) = dt * A12;
        A_sym.bottomLeftCorner(V_DIM, P_DIM) = dt * (A21 * (p_matrix_t::Identity() + dt * A11));
        A_sym.bottomRightCorner(V_DIM, V_DIM) = v_matrix_t::Identity() + dt * (A22 + dt * A21 * A12);

        // discrete approximation B matrix
        B_sym.topRows(P_DIM) = dt * B1;
        B_sym.bottomRows(V_DIM) = dt * (B2 + dt * A21 * B1);
    }


    //! gets instantiated in case the system is not symplectic
    SYMPLECTIC_DISABLED symplecticEuler(const StateVector<STATE_DIM, SCALAR>& x_n,
        const ControlVector<CONTROL_DIM, SCALAR>& u_n,
        const StateVector<STATE_DIM, SCALAR>& x_next,
        const int& n,
        state_matrix_t& A,
        state_control_matrix_t& B)
    {
        throw std::runtime_error("SensitivityApproximation : selected symplecticEuler but System is not symplectic.");
    }

    //! gets instantiated in case the system is not symplectic
    SYMPLECTIC_DISABLED symplecticEuler(const StateVector<STATE_DIM, SCALAR>& x,
        const ControlVector<CONTROL_DIM, SCALAR>& u,
        const int& n,
        state_matrix_t& A_sym,
        state_control_matrix_t& B_sym)
    {
        throw std::runtime_error("SensitivityApproximation : selected symplecticEuler but System is not symplectic.");
    }

    SYMPLECTIC_DISABLED getSymplecticEulerApproximation(const state_matrix_t& Ac1,
        const state_matrix_t& Ac2,
        const state_control_matrix_t& Bc1,
        const state_control_matrix_t& Bc2,
        state_matrix_t& A_sym,
        state_control_matrix_t B_sym)
    {
        throw std::runtime_error("SensitivityApproximation : selected symplecticEuler but System is not symplectic.");
    }

    //! shared_ptr to a continuous time linear system (system to be discretized)
    std::shared_ptr<quadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>> quadraticSystem_;

    //! discretization settings
    SensitivityQuadApproximationSettings settings_;
};


}  // namespace core
}  // namespace ct


#undef SYMPLECTIC_ENABLED
#undef SYMPLECTIC_DISABLED
